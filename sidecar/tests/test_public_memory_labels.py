"""The frozen old-value labels the memory benchmark's traps are scored against (``benchmarks/public_memory/labels.py``).

Synthetic datasets only: no dataset file, model or endpoint is read.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
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
def labels():
    return _load('colonymind_public_memory_labels', BENCH / 'labels.py')


@pytest.fixture(scope='module')
def records():
    return _load('colonymind_public_memory_records_for_labels', BENCH / 'datasets' / 'records.py')


def at(day, hour=9):
    return datetime(2023, 3, day, hour, 0, tzinfo=UTC)


def dataset(records, name='longmemeval_s'):
    """Two conversations: an office that moved (an update), a conflict over a start date (BEAM-shaped labels), an
    abstention question and a plain one."""
    Turn, Session, Evidence = records.Turn, records.Session, records.Evidence
    office = records.Conversation(id='c-1', sessions=(
        Session('s-a', 's-a', 0, at(1), (Turn(0, 'user', 'My office is in Room 4.'), Turn(1, 'assistant', 'Noted.'))),
        Session('s-b', 's-b', 1, at(9), (Turn(0, 'user', 'Lunch was good.'), Turn(1, 'assistant', 'Nice.'))),
        Session('s-c', 's-c', 2, at(20), (Turn(0, 'user', 'We moved: my office is Room 9 now.'),
                                          Turn(1, 'assistant', 'Updated.')))))
    start = records.Conversation(id='c-2', sessions=(
        Session('s-1', 's-1', 0, at(2), (Turn(0, 'user', 'I start on May 1.', ref='10'),)),
        Session('s-2', 's-2', 1, at(4), (Turn(0, 'user', 'I start on June 3.', ref='20'),))))
    questions = (
        records.Question('ku-1', 'c-1', 'knowledge-update', 'Where is my office?', 'Room 9', at(25), None, False,
                         ('s-a', 's-c'), (Evidence('s-a', 0), Evidence('s-c', 0))),
        records.Question('cr-1', 'c-2', 'contradiction_resolution', 'When do I start?', 'Ask which date holds.',
                         at(6), None, False, ('s-1', 's-2'),
                         (Evidence('s-1', 0, 'first_statement'), Evidence('s-2', 0, 'second_statement'))),
        records.Question('ku-2_abs', 'c-1', 'knowledge-update', 'Where is my car parked?', 'Not mentioned.', at(25),
                         None, True, ('s-c',), ()),
        records.Question('ss-1', 'c-1', 'single-session-user', 'What was lunch like?', 'Good', at(25), None, False,
                         ('s-b',), (Evidence('s-b', 0),)))
    return records.Dataset(name=name, sha256='0' * 64, conversations={'c-1': office, 'c-2': start},
                           questions=questions, findings={})


def test_each_question_has_one_trap_kind(labels, records):
    data = dataset(records)
    kinds = {question.id: labels.trap_kind(question) for question in data.questions}
    # An abstention question is an abstention trap whatever its type; a contradiction asks which side holds.
    assert kinds == {'ku-1': 'update', 'cr-1': 'conflict', 'ku-2_abs': 'abstain', 'ss-1': 'recall'}
    beam = records.Question('c:knowledge_update:0', 'c-1', 'knowledge_update', 'q', 'a', at(25), None, False, (), ())
    assert labels.trap_kind(beam) == 'update'
    assert labels.trap_kind(records.Question('c:abstention:0', 'c-1', 'abstention', 'q', 'a', at(25), None, True,
                                             (), ())) == 'abstain'


def test_the_draft_lists_the_evidence_to_label_and_labels_nothing_itself(labels, records):
    draft = labels.draft(dataset(records))
    assert draft['protocol'] == labels.DRAFT and draft['dataset'] == 'longmemeval_s'
    assert draft['dataset_sha256'] == '0' * 64
    entries = {entry['question']: entry for entry in draft['entries']}
    assert set(entries) == {'ku-1', 'cr-1'}
    update = entries['ku-1']
    assert update['kind'] == 'update' and update['values'] == [] and update['reviewed'] is False
    assert update['answer'] == 'Room 9' and update['text'] == 'Where is my office?'
    # The evidence in time order, the earlier turns marked: the old value is read from them by a reviewer.
    assert [(item['session'], item['position'], item['content']) for item in update['evidence']] == [
        ('s-a', 'earlier', 'My office is in Room 4.'), ('s-c', 'latest', 'We moved: my office is Room 9 now.')]
    conflict = entries['cr-1']
    assert conflict['kind'] == 'conflict' and conflict['values'] == []
    assert [item['label'] for item in conflict['evidence']] == ['first_statement', 'second_statement']


def reviewed(labels, records):
    draft = labels.draft(dataset(records))
    for entry in draft['entries']:
        entry['reviewed'] = True
        entry['values'] = ['Room 4'] if entry['kind'] == 'update' else ['May 1', 'June 3']
    return draft


def test_freezing_refuses_a_label_that_is_not_reviewed_or_not_well_formed(labels, records):
    data = dataset(records)

    def refused(change):
        draft = reviewed(labels, records)
        change({entry['question']: entry for entry in draft['entries']}, draft)
        with pytest.raises(labels.LabelError):
            labels.freeze(draft, data)

    refused(lambda entries, draft: entries['ku-1'].update(reviewed=False))
    refused(lambda entries, draft: entries['ku-1'].update(values=[]))
    refused(lambda entries, draft: entries['ku-1'].update(values=['  ']))
    # An old value that is the current answer would make the right answer a trap.
    refused(lambda entries, draft: entries['ku-1'].update(values=['room 9']))
    refused(lambda entries, draft: entries['cr-1'].update(values=['May 1']))
    refused(lambda entries, draft: entries['cr-1'].update(values=['May 1', 'may 1']))
    refused(lambda entries, draft: entries['cr-1'].update(kind='update'))
    refused(lambda entries, draft: draft['entries'].append(dict(entries['ku-1'], question='ku-9')))
    refused(lambda entries, draft: draft['entries'].append(dict(entries['ku-1'])))
    refused(lambda entries, draft: draft.update(dataset_sha256='1' * 64))
    refused(lambda entries, draft: draft.update(dataset='beam'))


def test_frozen_labels_are_pinned_by_hash_and_load_only_as_frozen(labels, records, tmp_path):
    data = dataset(records)
    draft = reviewed(labels, records)
    frozen, digest = labels.freeze(draft, data)
    # The same review in another order freezes to the same bytes.
    again, digest_again = labels.freeze({**draft, 'entries': list(reversed(draft['entries']))}, data)
    assert (frozen, digest) == (again, digest_again)
    path = tmp_path / 'old-values.json'
    path.write_bytes(frozen)
    loaded = labels.load(path, digest)
    assert loaded == {'ku-1': {'kind': 'update', 'type': 'knowledge-update', 'values': ['Room 4']},
                      'cr-1': {'kind': 'conflict', 'type': 'contradiction_resolution', 'values': ['May 1', 'June 3']}}
    # Only what the traps read is frozen: no evidence text, no review notes.
    assert 'evidence' not in json.loads(frozen)['entries'][0]
    with pytest.raises(labels.LabelError):
        labels.load(path, '0' * 64)
    path.write_bytes(frozen.replace(b'Room 4', b'Room 5'))
    with pytest.raises(labels.LabelError):
        labels.load(path, digest)


def _pinned(path, body):
    """A file of ``body`` in the canonical form freeze writes, and its hash: only the loader stands between it and
    a run."""
    data = (json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_a_draft_or_a_hand_made_file_never_loads_as_frozen_labels(labels, records, tmp_path):
    data = dataset(records)
    path = tmp_path / 'labels.json'
    # An untouched draft, pinned by its own hash: nothing reviewed, no old values. It is not a frozen file.
    draft = labels.draft(data)
    with pytest.raises(labels.LabelError):
        labels.load(path, _pinned(path, draft))
    # A reviewed draft is still a draft until freeze writes it.
    with pytest.raises(labels.LabelError):
        labels.load(path, _pinned(path, reviewed(labels, records)))
    frozen, digest = labels.freeze(reviewed(labels, records), data)
    body = json.loads(frozen)

    def refused(change, **options):
        changed = json.loads(frozen)
        change(changed, {entry['question']: entry for entry in changed['entries']})
        with pytest.raises(labels.LabelError):
            labels.load(path, _pinned(path, changed), **options)

    # Written as frozen but not well formed: every entry is checked as freeze checks it.
    refused(lambda body, entries: entries['ku-1'].update(values=[]))
    refused(lambda body, entries: entries['ku-1'].update(values=['  ']))
    refused(lambda body, entries: entries['cr-1'].update(values=['May 1']))
    refused(lambda body, entries: entries['cr-1'].update(values=['May 1', 'may 1']))
    refused(lambda body, entries: entries['ku-1'].update(kind='recall'))
    refused(lambda body, entries: entries['ku-1'].update(reviewed=True))
    refused(lambda body, entries: body['entries'].append(dict(entries['ku-1'])))
    refused(lambda body, entries: body['entries'].reverse())
    # Against the dataset a run reads: frozen from that dataset file, of its questions and kinds, no old value
    # that is the current answer.
    refused(lambda body, entries: None, dataset=dataset(records, name='beam'))
    refused(lambda body, entries: body.update(dataset_sha256='1' * 64), dataset=data)
    refused(lambda body, entries: entries['ku-1'].update(values=['room 9']), dataset=data)
    refused(lambda body, entries: entries['ku-1'].update(question='ku-9'), dataset=data)
    refused(lambda body, entries: entries['ku-1'].update(type='multi-session'), dataset=data)
    # The file freeze wrote loads, with or without the dataset beside it.
    path.write_bytes(frozen)
    assert labels.load(path, digest, dataset=data) == labels.load(path, digest)
    assert body['protocol'] == labels.FROZEN != labels.DRAFT


def test_the_command_line_drafts_freezes_and_checks(labels, records, tmp_path, monkeypatch, capsys):
    data = dataset(records)
    monkeypatch.setattr(labels, 'load_dataset', lambda name, data_dir, offline: data)
    draft_path, frozen_path = tmp_path / 'draft.json', tmp_path / 'frozen.json'
    assert labels.main(['draft', '--dataset', 'longmemeval_s', '--offline', '--out', str(draft_path)]) == 0
    draft = json.loads(draft_path.read_text())
    for entry in draft['entries']:
        entry.update(reviewed=True, values=['Room 4'] if entry['kind'] == 'update' else ['May 1', 'June 3'])
    draft_path.write_text(json.dumps(draft))
    capsys.readouterr()
    assert labels.main(['freeze', '--dataset', 'longmemeval_s', '--offline', '--draft', str(draft_path),
                        '--out', str(frozen_path)]) == 0
    digest = json.loads(capsys.readouterr().out)['sha256']
    assert labels.main(['check', '--labels', str(frozen_path), '--sha256', digest]) == 0
    assert labels.main(['check', '--labels', str(frozen_path), '--sha256', 'f' * 64]) == 1
