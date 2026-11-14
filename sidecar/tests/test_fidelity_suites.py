"""The fidelity suites: known traps from scripted models, graded trap first by a judge that is not the model under
test, over repeats; nothing is written by the borrowed-value suite, and the language suite runs the real night.

Every model here is scripted; nothing reads an endpoint.
"""
import asyncio
from contextlib import closing
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

FIDELITY = Path(__file__).resolve().parents[2] / 'benchmarks' / 'fidelity'
NOW = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)


def load(name):
    spec = importlib.util.spec_from_file_location(f'fidelity_{name}', FIDELITY / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def suites():
    return load('suites')


# ------------------------------------------------------------------ the sets

def test_each_set_is_frozen_by_its_hash(suites, tmp_path, monkeypatch):
    for name in suites.SETS:
        data = suites.load_set(name)
        assert data['items'] and data['version'].startswith('fidelity-')
    borrowed = suites.load_set('borrowed_value')
    for item in borrowed['items']:
        assert item['facts'] and item['expect'] and all(0 <= index < len(item['facts']) for index in item['expect'])
        assert item['text'].strip() and datetime.fromisoformat(item['at'])
    language = suites.load_set('language')
    assert any(item['language'] == 'English' for item in language['items'])
    for item in language['items']:
        assert len(item['statements']) == 3 and len(item['conversation']) >= 3
        assert all(row['value'] in row['text'] and row['subject'] in row['text'] for row in item['statements'])
    changed = tmp_path / 'borrowed_value.json'
    changed.write_text((FIDELITY / 'borrowed_value.json').read_text().replace('Denver', 'Dallas'))
    monkeypatch.setattr(suites, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='frozen set'):
        suites.load_set('borrowed_value')


def test_the_judge_is_never_the_model_under_test(suites):
    with pytest.raises(ValueError, match='not be the model under test'):
        suites.independent({'model': 'Model-A', 'base_url': 'http://a'}, {'model': 'model-a', 'base_url': 'http://b'})
    suites.independent({'model': 'model-a'}, {'model': 'model-b'})


# ------------------------------------------------------------------ grading, trap first

ITEM = {'id': 'x', 'text': 'My sister moved to Denver, but I still live in Boston.',
        'facts': [{'subject': 'my sister', 'value': 'Denver'}, {'subject': 'I', 'value': 'Boston'}], 'expect': [1]}
CLAIMS = [{'subject': 'I', 'predicate': 'lives in', 'value': 'Denver'},
          {'subject': 'I', 'predicate': 'lives in', 'value': 'Boston'}]


def test_a_borrowed_value_is_a_trap_and_a_trap_is_never_correct(suites):
    verdict = {'claims': [{'index': 0, 'holder': 'other'}, {'index': 1, 'holder': 'own'}],
               'facts': [{'index': 0, 'stated': False}, {'index': 1, 'stated': True}]}
    assert suites.grade_borrowed(ITEM, CLAIMS, verdict) == {
        'scored': True, 'trap': True, 'correct': False, 'borrowed': [0], 'missing': [], 'reason': None}
    clean = {'claims': [{'index': 0, 'holder': 'none'}, {'index': 1, 'holder': 'own'}],
             'facts': [{'index': 0, 'stated': False}, {'index': 1, 'stated': True}]}
    assert suites.grade_borrowed(ITEM, CLAIMS, clean)['correct'] is True
    missing = {**clean, 'facts': [{'index': 0, 'stated': True}, {'index': 1, 'stated': False}]}
    assert suites.grade_borrowed(ITEM, CLAIMS, missing) == {
        'scored': True, 'trap': False, 'correct': False, 'borrowed': [], 'missing': [1], 'reason': None}


FACTS = [{'index': 0, 'stated': True}, {'index': 1, 'stated': True}]
OWN = [{'index': 0, 'holder': 'own'}, {'index': 1, 'holder': 'own'}]


@pytest.mark.parametrize('verdict', [
    None, {}, {'claims': [{'index': 0, 'holder': 'own'}], 'facts': []},
    {'claims': [{'index': 0, 'holder': 'mine'}, {'index': 1, 'holder': 'own'}], 'facts': FACTS},
    {'claims': [{'index': 0, 'holder': 'own'}, {'index': '1', 'holder': 'own'}], 'facts': FACTS},
    {'claims': [{'index': 0, 'holder': 'own'}, {'holder': 'own'}], 'facts': FACTS},
    {'claims': [{'index': 0, 'holder': 'own'}, {'index': True, 'holder': 'own'}], 'facts': FACTS},
    {'claims': OWN, 'facts': [{'index': False, 'stated': True}, {'index': 1, 'stated': True}]},
    {'claims': OWN, 'facts': [{'index': 0, 'stated': 'yes'}, {'index': 1, 'stated': True}]},
    {'claims': OWN, 'facts': FACTS, 'note': 'extra'},
    {'claims': [{'index': 0, 'holder': 'own', 'why': 'x'}, {'index': 1, 'holder': 'own'}], 'facts': FACTS},
    {'claims': [{'index': 0, 'holder': 'own'}, {'index': 0, 'holder': 'own'}], 'facts': FACTS},
])
def test_a_judge_answer_out_of_schema_is_unscored_never_passed_and_never_aborts_the_run(suites, verdict):
    """Every verdict the borrowed-value schema does not admit, a wrong index type included, is unscored: it
    neither passes nor raises (a raise ended the run before its report was written)."""
    grade = suites.grade_borrowed(ITEM, CLAIMS, verdict)
    assert grade['scored'] is False and grade['trap'] is None and grade['reason'] == 'judge'
    assert suites.grade_language('Un texto.', verdict)['scored'] is False


@pytest.mark.parametrize('verdict', [
    {'same_language': True, 'faithful': True},
    {'source_language': 'Spanish', 'same_language': True, 'faithful': True},
    {'source_language': 'Spanish', 'output_language': 3, 'same_language': True, 'faithful': True},
    {'source_language': 'Spanish', 'output_language': 'Spanish', 'same_language': 1, 'faithful': True},
    {'source_language': 'Spanish', 'output_language': 'Spanish', 'same_language': True, 'faithful': True,
     'note': 'extra'},
])
def test_a_language_verdict_missing_or_mistyping_a_field_is_unscored(suites, verdict):
    grade = suites.grade_language('Es enfermera.', verdict)
    assert grade['scored'] is False and grade['reason'] == 'judge'


def test_the_schemas_the_judge_is_given_are_the_ones_its_answers_are_checked_against(suites):
    whole = {'claims': OWN, 'facts': FACTS}
    assert suites.fits(whole, suites.BORROWED_SCHEMA) and suites.grade_borrowed(ITEM, CLAIMS, whole)['scored']
    language = {'source_language': 'Spanish', 'output_language': 'Spanish', 'same_language': True, 'faithful': True}
    assert suites.fits(language, suites.LANGUAGE_SCHEMA)
    assert suites.grade_language('Es enfermera.', language)['scored'] is True


def test_a_text_in_another_language_is_a_trap(suites):
    assert suites.grade_language('She is a nurse.', {'source_language': 'Spanish', 'output_language': 'English',
                                                     'same_language': False, 'faithful': True})['trap'] is True
    kept = suites.grade_language('Es enfermera.', {'source_language': 'Spanish', 'output_language': 'Spanish',
                                                   'same_language': True, 'faithful': True})
    assert kept['trap'] is False and kept['correct'] is True
    assert suites.grade_language('', {'same_language': True, 'faithful': True})['scored'] is False


def test_the_summary_counts_traps_over_every_scored_attempt(suites):
    rows = [{'id': 'a', 'grade': {'scored': True, 'trap': True, 'correct': False}},
            {'id': 'a', 'grade': {'scored': True, 'trap': False, 'correct': True}},
            {'id': 'b', 'grade': {'scored': True, 'trap': False, 'correct': True}},
            {'id': 'b', 'grade': {'scored': False, 'trap': None, 'correct': None}}]
    summary = suites.summarize(rows)
    assert (summary['questions'], summary['attempts'], summary['scored'], summary['unscored']) == (2, 4, 3, 1)
    assert summary['traps'] == 1 and summary['questions_with_a_trap'] == ['a']
    assert summary['trap_rate']['successes'] == 1 and summary['trap_rate']['trials'] == 3
    assert summary['correct_rate']['successes'] == 2
    assert summary['by_question']['a'] == {'attempts': 2, 'scored': 2, 'traps': 1, 'correct': 1}


# ------------------------------------------------------------------ borrowed value through the dry run

class Extractor:
    """The model under test, scripted: the message's claims as given, and review keeps them."""

    model = 'model-under-test'

    def __init__(self, claims):
        self.claims, self.calls = claims, 0

    def tier_config(self, tier):
        return SimpleNamespace(base_url='http://127.0.0.1:8080/v1')

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if (kwargs.get('context') or {}).get('task') == 'source_claim_review':
            return SimpleNamespace(content=json.dumps({str(row['index']): {'keep': True, 'reason': 'kept'}
                                                       for row in payload['proposals']}), model_id=self.model)
        self.calls += 1
        rows = [{'subject': subject, 'predicate': predicate, 'value': value, 'evidence': payload['message'],
                 'operation': 'assert', 'prior_claim_id': None, 'memory_kind': 'personal_context',
                 'recall_reason': 'Use it when replying.', 'valid_from_text': None, 'valid_to_text': None,
                 'event_at_text': None} for subject, predicate, value in self.claims]
        return SimpleNamespace(content=json.dumps(rows), model_id=self.model)


class ScriptedJudge:
    """The judge's router, scripted: labels each claim by the value it names, and each fact stated."""

    supports_function_routing = True

    def __init__(self, holders=None, *, language=None):
        self.holders, self.language, self.calls = holders or {}, language, []

    async def complete(self, messages=None, *, context=None, **kwargs):
        self.calls.append((context or {}).get('task'))
        payload = json.loads(messages[-1]['content'])
        if 'claims' in payload:
            claims = [{'index': row['index'], 'holder': self.holders.get(row['value'], 'none')}
                      for row in payload['claims']]
            values = {(row['subject'], row['value']) for row in payload['claims']}
            facts = [{'index': row['index'], 'stated': (row['subject'], row['value']) in values}
                     for row in payload['facts']]
            return SimpleNamespace(content=json.dumps({'claims': claims, 'facts': facts}))
        output = self.language(payload['text']) if self.language else 'Spanish'
        return SimpleNamespace(content=json.dumps({'source_language': 'Spanish', 'output_language': output,
                                                   'same_language': output == 'Spanish', 'faithful': True}))


def dry_run(suites, tmp_path, claims, holders):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    judge = suites.Judge(ScriptedJudge(holders))
    rows = asyncio.run(suites.borrowed_attempt(ITEM | {'at': '2026-05-04T10:00:00+00:00'},
                                               projection=SourceClaimProjection(ledger), router=Extractor(claims),
                                               judge=judge, contact_id='p-owner'))
    with closing(ledger._connect()) as conn:
        written = {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                   for table in ('turn_sources', 'source_claims', 'source_claim_jobs')}
    return rows, written, judge


def test_the_borrowed_value_suite_finds_a_known_trap_through_the_dry_run_and_writes_nothing(suites, tmp_path):
    row, written, judge = dry_run(suites, tmp_path, [('I', 'lives in', 'Denver'), ('I', 'lives in', 'Boston')],
                                  {'Denver': 'other', 'Boston': 'own'})
    assert row['status'] == 'complete' and [claim['value'] for claim in row['claims']] == ['Denver', 'Boston']
    assert row['grade'] == {'scored': True, 'trap': True, 'correct': False, 'borrowed': [0], 'missing': [],
                            'reason': None}
    assert written == {'turn_sources': 0, 'source_claims': 0, 'source_claim_jobs': 0}
    assert judge.calls == 1


def test_the_borrowed_value_suite_passes_a_message_kept_apart(suites, tmp_path):
    row, _, _ = dry_run(suites, tmp_path, [('my sister', 'lives in', 'Denver'), ('I', 'lives in', 'Boston')],
                        {'Denver': 'own', 'Boston': 'own'})
    assert row['grade'] == {'scored': True, 'trap': False, 'correct': True, 'borrowed': [], 'missing': [],
                            'reason': None}


def test_an_extraction_with_no_claim_is_no_trap_and_not_correct(suites, tmp_path):
    row, _, judge = dry_run(suites, tmp_path, [], {})
    assert row['grade']['trap'] is False and row['grade']['correct'] is False and row['grade']['missing'] == [1]
    assert judge.calls == 0


# ------------------------------------------------------------------ language through the real night

class NightModel:
    """The model under test for the night and the plain summary, scripted: writes in ``language``, its JSON fenced
    when ``fenced`` (production's parsers accept it)."""

    supports_function_routing = True

    def __init__(self, write, *, fenced=False):
        self.write, self.tasks, self.fenced = write, [], fenced

    def function_deadline_seconds(self, *, context=None):
        return 20

    async def complete(self, messages=None, *, context=None, **kwargs):
        from colonymind.mind.consolidate import TASK_DIGEST, TASK_EPISODE
        task = (context or {}).get('task')
        self.tasks.append(task)
        prompt = messages[-1]['content']
        if task == TASK_DIGEST:
            ids = sorted(set(re.findall(r'claim:[0-9a-f]{64}', prompt)))
            answer = {'digest': self.write('digest'), 'sources': ids}
        elif task == TASK_EPISODE:
            answer = {'summary': self.write('episode')}
        else:
            answer = {'sentence': self.write('summary')}
        text = json.dumps(answer, ensure_ascii=False)
        return SimpleNamespace(content=f'```json\n{text}\n```' if self.fenced else text, usage={'total_tokens': 10})


SPANISH = {'digest': 'Sam trabaja como enfermera y prefiere mensajes por la tarde.',
           'episode': 'Organizaron la cena del sábado para ocho personas.',
           'summary': 'Quiere que le recuerdes llamar al casero el lunes.'}
ENGLISH = {'digest': 'Sam works as a nurse and prefers messages in the afternoon.',
           'episode': 'They planned Saturday dinner for eight people.',
           'summary': 'Remind them to call the landlord on Monday.'}


@pytest.mark.parametrize('written, traps', [(SPANISH, 0), (ENGLISH, 3)])
def test_the_language_suite_runs_the_night_and_finds_a_translation(suites, tmp_path, monkeypatch, written, traps):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', suites.OWNER)
    item = next(row for row in suites.load_set('language')['items'] if row['id'] == 'lang-es')
    model = NightModel(written.get)
    judge = suites.Judge(ScriptedJudge(language=lambda text: 'Spanish' if text in SPANISH.values() else 'English'))
    rows = asyncio.run(suites.language_attempt(item, router=model, judge=judge, work=tmp_path / 'attempt', now=NOW))
    from colonymind.mind import plain
    from colonymind.mind.consolidate import TASK_DIGEST, TASK_EPISODE
    assert set(model.tasks) == {TASK_DIGEST, TASK_EPISODE, plain.SUMMARY_TASK}   # nothing else spent the model
    assert [row['kind'] for row in rows] == ['digest', 'episode', 'summary']
    assert [row['output'] for row in rows] == [written['digest'], written['episode'], written['summary']]
    assert sum(row['grade']['trap'] for row in rows) == traps
    assert all(row['grade']['scored'] for row in rows) and judge.calls == 3


def test_an_output_the_model_never_wrote_is_unscored(suites, tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', suites.OWNER)
    item = next(row for row in suites.load_set('language')['items'] if row['id'] == 'lang-de')
    judge = suites.Judge(ScriptedJudge())
    rows = asyncio.run(suites.language_attempt(item, router=NightModel(lambda kind: ''), judge=judge,
                                               work=tmp_path / 'attempt', now=NOW))
    assert [row['grade']['reason'] for row in rows] == ['no_output'] * 3 and judge.calls == 0


def test_the_runner_repeats_each_question_and_refuses_a_judge_that_is_the_model(suites, tmp_path, monkeypatch):
    run = load('run')
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', suites.OWNER)
    with pytest.raises(ValueError, match='not be the model under test'):
        run.check_endpoints({'CHAT_MODEL': 'same', 'JUDGE_MODEL': 'same'})
    items = suites.load_set('borrowed_value')['items'][:2]
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    projection = SourceClaimProjection(TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db'))
    report = asyncio.run(run.borrowed(items, projection=projection,
                                      router=Extractor([('I', 'lives in', 'Boston')]),
                                      judge=suites.Judge(ScriptedJudge({'Boston': 'own'})), repeats=3))
    assert report['summary']['attempts'] == 6 and report['summary']['questions'] == 2
    assert [row['repeat'] for row in report['rows']] == [1, 1, 2, 2, 3, 3]


def test_what_is_graded_is_what_production_accepted_never_a_rejected_answer(suites, tmp_path, monkeypatch):
    """A summary the plain summary's no-cut check refuses (here one that trails off in an ellipsis) never reaches
    the owner, so it is never graded: the attempt is unscored, and the row says the model answered."""
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', suites.OWNER)
    item = next(row for row in suites.load_set('language')['items'] if row['id'] == 'lang-ja')
    refused = {'digest': 'Sam works as a nurse.', 'episode': 'They planned dinner for eight.',
               'summary': '\u6628\u65e5\u5927\u5bb6\u3055\u3093\u3068\u8a71\u3057\u307e\u3057\u305f\u2026\u2026'}
    judge = suites.Judge(ScriptedJudge(language=lambda text: 'Japanese'))
    rows = asyncio.run(suites.language_attempt(item, router=NightModel(refused.get), judge=judge,
                                               work=tmp_path / 'attempt', now=NOW))
    summary = next(row for row in rows if row['kind'] == 'summary')
    assert summary['output'] is None and summary['answered'] == 1
    assert summary['grade']['scored'] is False and summary['grade']['reason'] == 'no_output'
    assert judge.calls == 2                                          # the digest and the episode only


def test_a_fenced_answer_production_accepts_is_graded(suites, tmp_path, monkeypatch):
    """The night and the plain summary parse a fenced JSON answer; the suite grades what they stored and returned,
    so a translation in a fenced answer is found."""
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', suites.OWNER)
    item = next(row for row in suites.load_set('language')['items'] if row['id'] == 'lang-es')
    judge = suites.Judge(ScriptedJudge(language=lambda text: 'Spanish' if text in SPANISH.values() else 'English'))
    rows = asyncio.run(suites.language_attempt(item, router=NightModel(ENGLISH.get, fenced=True), judge=judge,
                                               work=tmp_path / 'attempt', now=NOW))
    assert [row['output'] for row in rows] == [ENGLISH['digest'], ENGLISH['episode'], ENGLISH['summary']]
    assert all(row['grade']['scored'] and row['grade']['trap'] for row in rows) and judge.calls == 3


def test_the_graded_digest_and_episode_are_the_ones_the_night_stored(suites, tmp_path, monkeypatch):
    """A digest the night rejects (it cites none of the person's claims) is not stored, so it is not graded."""
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', suites.OWNER)
    item = next(row for row in suites.load_set('language')['items'] if row['id'] == 'lang-es')

    class Uncited(NightModel):
        async def complete(self, messages=None, *, context=None, **kwargs):
            from colonymind.mind.consolidate import TASK_DIGEST
            if (context or {}).get('task') == TASK_DIGEST:
                self.tasks.append(TASK_DIGEST)
                return SimpleNamespace(content=json.dumps({'digest': SPANISH['digest'], 'sources': []}),
                                       usage={'total_tokens': 10})
            return await super().complete(messages, context=context, **kwargs)
    judge = suites.Judge(ScriptedJudge())
    rows = asyncio.run(suites.language_attempt(item, router=Uncited(SPANISH.get), judge=judge,
                                               work=tmp_path / 'attempt', now=NOW))
    digest = rows[0]
    assert digest['kind'] == 'digest' and digest['output'] is None and digest['answered'] == 1
    assert digest['grade']['scored'] is False and [row['output'] for row in rows[1:]] == [
        SPANISH['episode'], SPANISH['summary']]
