"""The borrowed-value suite across messages, and the report of subjects named from earlier messages in formed stores.

A conversation's earlier messages are recorded for real and their claims formed by the model under test; its last
message goes through the claim dry run, which offers them. A claim whose value is another subject's, or a subject the
conversation leaves open, is a trap, graded first by a judge that is not the model under test. Every model here is
scripted; nothing reads an endpoint.
"""
import asyncio
from contextlib import closing
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

FIDELITY = Path(__file__).resolve().parents[2] / 'benchmarks' / 'fidelity'


def load(name):
    spec = importlib.util.spec_from_file_location(f'fidelity_{name}', FIDELITY / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def suites():
    return load('suites')


# ------------------------------------------------------------------ the set

def test_the_conversation_set_is_frozen_and_well_formed(suites, tmp_path, monkeypatch):
    data = suites.load_set('borrowed_value_context')
    assert data['version'] == 'fidelity-borrowed-value-context-1' and len(data['items']) == 16
    assert len({item['id'] for item in data['items']}) == 16
    for item in data['items']:
        said = [datetime.fromisoformat(row['at']) for row in item['earlier']] + [datetime.fromisoformat(item['at'])]
        assert item['earlier'] and said == sorted(said) and len(set(said)) == len(said)
        assert all(row['text'].strip() for row in item['earlier']) and item['text'].strip()
        assert item['facts'] and all(0 <= index < len(item['facts']) for index in item['expect'])
        if item.get('control') in {'ambiguous antecedent', 'no antecedent'}:
            assert item['expect'] == []
        else:
            assert item['expect']
    assert {item.get('control') for item in data['items']} == {
        None, 'ambiguous antecedent', 'no antecedent', 'subject in the message'}
    # The one-message set is the one its pre-registration froze.
    raw = (FIDELITY / 'borrowed_value.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == suites.BORROWED_SHA256 == (
        '9bbf702c4e3f54cda31577db307651492e6599da7337389c33f78434895dac92')
    changed = tmp_path / 'borrowed_value_context.json'
    changed.write_text((FIDELITY / 'borrowed_value_context.json').read_text().replace('Denver', 'Dallas'))
    monkeypatch.setattr(suites, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='frozen set'):
        suites.load_set('borrowed_value_context')


# ------------------------------------------------------------------ grading, trap first

ITEM = {'id': 'x', 'facts': [{'subject': 'my brother', 'value': 'a job at the hospital'},
                             {'subject': 'my sister', 'value': 'Denver'}], 'expect': [0]}
CLAIMS = [{'subject': 'my sister', 'predicate': 'job', 'value': 'a job at the hospital'},
          {'subject': 'my brother', 'predicate': 'job', 'value': 'a job at the hospital'}]
FACTS = [{'index': 0, 'stated': True}, {'index': 1, 'stated': False}]


@pytest.mark.parametrize('holders, trap, correct', [
    (['other', 'own'], True, False),
    (['ambiguous', 'own'], True, False),
    (['none', 'own'], False, True),
    (['own', 'own'], False, True),
])
def test_another_subject_s_value_or_an_open_one_is_a_trap_and_never_correct(suites, holders, trap, correct):
    verdict = {'claims': [{'index': index, 'holder': holder} for index, holder in enumerate(holders)],
               'facts': FACTS}
    grade = suites.grade_context(ITEM, CLAIMS, verdict)
    assert (grade['scored'], grade['trap'], grade['correct']) == (True, trap, correct)
    assert grade['borrowed'] == [index for index, holder in enumerate(holders) if holder == 'other']
    assert grade['ambiguous'] == [index for index, holder in enumerate(holders) if holder == 'ambiguous']


@pytest.mark.parametrize('verdict', [
    None, {}, {'claims': [{'index': 0, 'holder': 'mine'}, {'index': 1, 'holder': 'own'}], 'facts': FACTS},
    {'claims': [{'index': 0, 'holder': 'own'}], 'facts': FACTS},
    {'claims': [{'index': 0, 'holder': 'own'}, {'index': 1, 'holder': 'own'}], 'facts': FACTS[:1]},
])
def test_a_judge_answer_out_of_schema_is_unscored(suites, verdict):
    grade = suites.grade_context(ITEM, CLAIMS, verdict)
    assert grade['scored'] is False and grade['trap'] is None and grade['reason'] == 'judge'


# ------------------------------------------------------------------ through the production path

class Extractor:
    """The model under test, scripted: each message's proposals (``basis``: the earlier text and the words naming the
    subject, resolved against the offered earlier messages); review keeps everything. Extraction calls are kept."""

    model = 'model-under-test'

    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []

    def tier_config(self, tier):
        return SimpleNamespace(base_url='http://127.0.0.1:8080/v1')

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if (kwargs.get('context') or {}).get('task') == 'source_claim_review':
            return SimpleNamespace(content=json.dumps({str(row['index']): {'keep': True, 'reason': 'kept'}
                                                       for row in payload['proposals']}), model_id=self.model)
        self.calls.append(payload)
        rows = []
        for subject, predicate, value, basis in self.outputs.get(payload['message'], []):
            row = {'subject': subject, 'predicate': predicate, 'value': value, 'evidence': payload['message'],
                   'operation': 'assert', 'prior_claim_id': None, 'memory_kind': 'personal_context',
                   'recall_reason': 'Use it when replying.', 'valid_from_text': None, 'valid_to_text': None,
                   'event_at_text': None}
            if basis:
                text, words = basis
                row['subject_basis'] = {'message_id': next(entry['id'] for entry in payload['earlier_messages']
                                                           if entry['text'] == text), 'quote': words}
            rows.append(row)
        return SimpleNamespace(content=json.dumps(rows), model_id=self.model)


class ScriptedJudge:
    """The judge, scripted: each claim's holder by its subject and value; a fact is stated by a claim of both."""

    supports_function_routing = True

    def __init__(self, holders):
        self.holders, self.payloads = holders, []

    async def complete(self, messages=None, *, context=None, **kwargs):
        payload = json.loads(messages[-1]['content'])
        self.payloads.append((messages[0]['content'], payload))
        claims = [{'index': row['index'], 'holder': self.holders.get((row['subject'], row['value']), 'none')}
                  for row in payload['claims']]
        stated = {(row['subject'].casefold(), row['value']) for row in payload['claims']}
        facts = [{'index': row['index'], 'stated': (row['subject'].casefold(), row['value']) in stated}
                 for row in payload['facts']]
        return SimpleNamespace(content=json.dumps({'claims': claims, 'facts': facts}))


def bait(suites):
    return next(item for item in suites.load_set('borrowed_value_context')['items'] if item['id'] == 'bvc-06')


def counts(ledger):
    with closing(ledger._connect()) as conn:
        return {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                for table in ('turn_sources', 'source_claims', 'source_claim_jobs')}


@pytest.mark.parametrize('named, holders, trap', [
    ('my brother', {('my brother', 'a job at the hospital'): 'own'}, False),
    ('My sister', {('My sister', 'a job at the hospital'): 'other'}, True),
])
def test_the_conversation_suite_runs_the_setup_and_the_dry_run_and_finds_a_borrowed_value(
        suites, tmp_path, named, holders, trap):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    item = bait(suites)
    earlier = item['earlier'][0]['text']
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    model = Extractor({earlier: [('My sister', 'location', 'Denver', None), ('my brother', 'location', 'Austin', None)],
                       item['text']: [(named, 'job', 'a job at the hospital', (earlier, named))]})
    setup = asyncio.run(suites.context_setup(item, ledger=ledger, router=model, contact_id='p-owner'))
    assert setup == {'messages': 1, 'jobs': ['complete'], 'claims': 2}
    with closing(ledger._connect()) as conn:
        person, session = conn.execute('SELECT contact_id,session_id FROM turn_sources').fetchone()
    assert (person, session) == ('p-owner-bvc-06', 'fidelity-context-bvc-06')
    before = counts(ledger)
    judge = ScriptedJudge(holders)
    row = asyncio.run(suites.context_attempt(item, projection=SourceClaimProjection(ledger), router=model,
                                             judge=suites.Judge(judge), contact_id='p-owner'))
    assert counts(ledger) == before                                    # the dry run writes nothing
    assert model.calls[-1]['earlier_messages'][0]['text'] == earlier  # the dry run offered the earlier message
    assert row['status'] == 'complete' and row['offered_earlier'] == 1 and row['named_from_earlier'] == [named]
    assert row['claims'] == [{'subject': named, 'predicate': 'job', 'value': 'a job at the hospital'}]
    assert row['grade']['trap'] is trap and row['grade']['correct'] is (not trap)
    system, payload = judge.payloads[0]
    assert system == suites.CONTEXT_SYSTEM
    assert payload['earlier_messages'] == [earlier] and payload['last_message'] == item['text']


def test_an_ambiguous_antecedent_named_anyway_is_a_trap_and_no_claim_is_none(suites, tmp_path):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    item = next(row for row in suites.load_set('borrowed_value_context')['items'] if row['id'] == 'bvc-05')
    earlier = item['earlier'][0]['text']
    for name, outputs, trap in (('named', [('My sister', 'moving to', 'Lisbon', (earlier, 'My sister'))], True),
                                ('abstained', [], False)):
        (tmp_path / name).mkdir()
        ledger = TurnIdempotencyLedger(tmp_path / name / 'turn-idempotency.db')
        model = Extractor({item['text']: outputs})
        asyncio.run(suites.context_setup(item, ledger=ledger, router=model, contact_id='p-owner'))
        judge = suites.Judge(ScriptedJudge({('My sister', 'Lisbon'): 'ambiguous'}))
        row = asyncio.run(suites.context_attempt(item, projection=SourceClaimProjection(ledger), router=model,
                                                 judge=judge, contact_id='p-owner'))
        assert row['grade']['trap'] is trap
        assert row['grade']['correct'] is (not trap)                  # the control expects no claim
        assert judge.calls == (1 if outputs else 0)


def test_the_runner_sets_each_conversation_up_once_and_repeats_its_last_message(suites, tmp_path):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    run = load('run')
    items = suites.load_set('borrowed_value_context')['items'][:2]
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    model = Extractor({})
    report = asyncio.run(run.borrowed_context(items, ledger=ledger, projection=SourceClaimProjection(ledger),
                                              router=model, judge=suites.Judge(ScriptedJudge({})), repeats=3))
    assert [row['repeat'] for row in report['rows']] == [1, 1, 2, 2, 3, 3]
    assert report['summary']['attempts'] == 6 and report['summary']['traps'] == 0
    assert set(report['setup']) == {'bvc-01', 'bvc-02'}
    assert all(setup['jobs'] == ['complete'] for setup in report['setup'].values())
    with closing(ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 2   # recorded once, not per repeat


# ------------------------------------------------------------------ subjects named in formed stores

MOVED, LIVES = 'My sister moved last month.', 'She lives in Denver now.'


def formed(root, *, named=True):
    """A store in the replay's layout whose conversation formed one claim, named from an earlier message or not."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    (root / 'instance').mkdir(parents=True)
    (root / 'replay.json').write_text(json.dumps({'questions': ['q-' + root.name]}))
    ledger = TurnIdempotencyLedger(root / 'instance' / 'turn-idempotency.db')
    for turn, text, minute in (('moved', MOVED, 0), ('lives', LIVES, 1)):
        ledger.record_source(turn, contact_id='owner', session_id='s', occurred_at=f'2026-05-04T10:0{minute}:00+00:00',
                             messages=[{'role': 'user', 'content': text}])
    subject, basis = ('My sister', (MOVED, 'My sister')) if named else ('She', None)
    model = Extractor({LIVES: [(subject, 'location', 'Denver', basis)]})
    projection = SourceClaimProjection(ledger)
    while asyncio.run(projection.process_one(model)):
        pass
    return root


def test_the_report_lists_the_claims_named_from_earlier_messages_and_writes_nothing(tmp_path):
    report_module = load('subject_bases')
    formed(tmp_path / 'stores' / 'a')
    formed(tmp_path / 'stores' / 'b', named=False)
    found = report_module.stores([tmp_path / 'stores'])
    assert [row['name'] for row in found] == ['a', 'b'] and found[0]['questions'] == ['q-a']
    ledger = found[0]['ledger']
    before = ledger.read_bytes()
    read = [report_module.read_store(row) for row in found]
    assert ledger.read_bytes() == before
    first, second = read
    assert first['claims'] == 1 and first['by_version'] == {'source-claims-v19': 1}
    row, = first['named']
    assert (row['subject'], row['value'], row['named_by']) == ('My sister', 'Denver', 'My sister')
    assert (row['earlier_message'], row['message'], row['live'], row['basis_stands']) == (MOVED, LIVES, True, True)
    assert second['claims'] == 1 and second['named'] == []
    report = {'stores': read}
    judge = ScriptedJudge({('My sister', 'Denver'): 'own'})
    asyncio.run(report_module.judge_named(report, load('suites').Judge(judge)))
    assert row['grade'] == {'scored': True, 'holder': 'own', 'trap': False, 'correct': True}
    assert row['conversation'] == [{'role': 'user', 'text': MOVED}]
    assert judge.payloads[0][1]['earlier_messages'] == row['conversation']
    assert judge.payloads[0][1]['last_message'] == LIVES
    summary = report_module.summarize(report)
    assert (summary['stores'], summary['claims'], summary['named'], summary['judged'], summary['traps']) == (
        2, 2, 1, 1, 0)
    assert summary['correct_rate']['successes'] == 1
    with pytest.raises(ValueError, match='not a store'):
        report_module.stores([tmp_path / 'nothing-here'])


def test_the_report_reads_a_store_of_the_build_before_with_that_build(tmp_path, monkeypatch):
    """The comparison runs the report with each build's own interpreter, and the build before this version has no
    reader of a subject named from an earlier message (``message_subject_basis``, the offered earlier messages): its
    stores, which hold no such claim, are read and counted all the same."""
    from colonymind.beliefs import source_projection
    report_module = load('subject_bases')
    with monkeypatch.context() as patch:
        patch.setattr(source_projection, 'EXTRACTION_VERSION', 'source-claims-v18')
        formed(tmp_path / 'before', named=False)
    monkeypatch.delattr(source_projection, 'message_subject_basis')
    monkeypatch.delattr(source_projection.SourceClaimProjection, 'earlier_messages')
    store, = report_module.stores([tmp_path / 'before'])
    read = report_module.read_store(store)
    assert (read['claims'], read['by_version'], read['named']) == (1, {'source-claims-v18': 1}, [])
    report = {'stores': [read]}
    asyncio.run(report_module.judge_named(report, load('suites').Judge(ScriptedJudge({}))))
    assert (report_module.summarize(report)['claims'], report_module.summarize(report)['named']) == (1, 0)


MOTHER = 'My mother is staying with us this week.'


def test_the_judge_reads_the_conversation_between_the_basis_and_the_claim(tmp_path):
    """A message naming another subject between the basis and the claim's message (here the mother) is what makes
    "She lives in Denver now." hers or open: the judge reads the conversation from the first earlier message the
    extraction was offered to the claim's own message, the assistant's replies too, never the basis alone."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import TurnIdempotencyLedger
    report_module = load('subject_bases')
    root = tmp_path / 'store'
    (root / 'instance').mkdir(parents=True)
    ledger = TurnIdempotencyLedger(root / 'instance' / 'turn-idempotency.db')
    ledger.record_source('before', contact_id='owner', session_id='other', occurred_at='2026-05-04T09:59:00+00:00',
                         messages=[{'role': 'user', 'content': 'My aunt lives in Ohio.'}])
    for turn, messages, minute in (
            ('moved', [{'role': 'user', 'content': MOVED}, {'role': 'assistant', 'content': 'How is your mother?'}], 0),
            ('mother', [{'role': 'user', 'content': MOTHER}], 1),
            ('lives', [{'role': 'user', 'content': LIVES}, {'role': 'assistant', 'content': 'Noted.'}], 2)):
        ledger.record_source(turn, contact_id='owner', session_id='s', occurred_at=f'2026-05-04T10:0{minute}:00+00:00',
                             messages=messages)
    projection = SourceClaimProjection(ledger)
    model = Extractor({LIVES: [('My sister', 'location', 'Denver', (MOVED, 'My sister'))]})
    while asyncio.run(projection.process_one(model)):
        pass
    store, = report_module.stores([root])
    read = report_module.read_store(store)
    row, = read['named']
    assert row['conversation'] == [{'role': 'user', 'text': MOVED},
                                   {'role': 'assistant', 'text': 'How is your mother?'},
                                   {'role': 'user', 'text': MOTHER}]
    judge = ScriptedJudge({('My sister', 'Denver'): 'ambiguous'})
    report = {'stores': [read]}
    asyncio.run(report_module.judge_named(report, load('suites').Judge(judge)))
    system, payload = judge.payloads[0]
    assert payload['earlier_messages'] == row['conversation'] and payload['last_message'] == LIVES
    assert '"assistant"' in system and system != load('suites').CONTEXT_SYSTEM
    assert row['grade'] == {'scored': True, 'holder': 'ambiguous', 'trap': True, 'correct': False}
    assert report_module.summarize(report)['traps'] == 1
    # A claim of another conversation that keeps this basis (a correction carries it) is judged in its own
    # conversation, the basis message first.
    fix = 'Correction: she lives in Dallas.'
    ledger.record_source('later', contact_id='owner', session_id='other', occurred_at='2026-05-04T10:05:00+00:00',
                         messages=[{'role': 'user', 'content': fix}])
    from colonymind.turns.idempotency import source_message_hash
    uri = store['ledger'].resolve().as_uri() + '?mode=ro'
    with closing(report_module.sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = report_module.sqlite3.Row
        judged_in = report_module.conversation(conn, uri, 'later', source_message_hash('other', {
            'role': 'user', 'content': fix}), {'turn_id': 'moved', 'message_hash': source_message_hash('s', {
                'role': 'user', 'content': MOVED})})
    assert judged_in == [{'role': 'user', 'text': MOVED}, {'role': 'user', 'text': 'My aunt lives in Ohio.'}]
