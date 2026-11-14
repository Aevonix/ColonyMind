"""How a formed store holds a question's changed or disputed value, read from the store's own links, never from text.

For a knowledge-update or contradiction question, the claims its evidence messages yielded (anchored at one, or
witnessed by one) are classified in this order: an old value replaced or withdrawn (``superseded``); two live values
the night decided cannot both stand (``conflict``: the store keeps both and asks); two different live values, each
from a different evidence message, with nothing settling them (``both_current``); one live value however often it is
stated, or values one evidence message alone states (``single_value``); nothing stored from the evidence
(``not_stored``).
"""
from contextlib import closing
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from test_public_memory_replay import formed, records, replay_module  # noqa: F401  (pytest fixtures)

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
def storage():
    return _load('colonymind_public_memory_storage', BENCH / 'storage.py')


def question(records, kind='knowledge-update', evidence=(('s-1', 0), ('s-2', 0)), qid='q-1', abstention=False):
    return records.Question(qid, 'c-1', kind, 'Where is my office now?', 'Room 9',
                            datetime(2023, 4, 1, 12, tzinfo=UTC), None, abstention, ('s-2',),
                            tuple(records.Evidence(session, turn) for session, turn in evidence))


def store(root, claims, *, decided=(), witnesses=(), kind='formed', turns=None):
    """A store directory as a replay leaves it: the manifest's map from dataset turns to stored messages, the
    ledger's claims and witnesses, and the night's decided relations in ``mind.db``."""
    from colonymind.mind.concerns import MindState, open_mind_db
    from colonymind.mind.consolidate import RELATION_KEY, _pair_id
    from colonymind.turns import TurnIdempotencyLedger
    home = root / ('before-night' if kind == 'before_night' else '') / 'instance'
    home.mkdir(parents=True, exist_ok=True)
    turns = turns if turns is not None else [
        {'session': 's-1', 'index': 0, 'role': 'user', 'turn_id': 't-1', 'message_hash': 'h-1', 'alone': False},
        {'session': 's-1', 'index': 1, 'role': 'assistant', 'turn_id': 't-1', 'message_hash': 'h-1a',
         'alone': False},
        {'session': 's-2', 'index': 0, 'role': 'user', 'turn_id': 't-2', 'message_hash': 'h-2', 'alone': False},
        {'session': 's-3', 'index': 0, 'role': 'user', 'turn_id': 't-3', 'message_hash': 'h-3', 'alone': True}]
    manifest = root / 'replay.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({'protocol': 'public-memory-replay-1',
                                        'store': 'import_only' if kind == 'import_only' else 'formed',
                                        'turns': turns}))
    TurnIdempotencyLedger(home / 'turn-idempotency.db')
    with closing(sqlite3.connect(home / 'turn-idempotency.db')) as conn, conn:
        for row in claims:
            conn.execute('INSERT INTO source_claims(id,turn_id,message_hash,subject_key,predicate,value_key,data_json,'
                         'superseded_by,retracted_by) VALUES (?,?,?,?,?,?,?,?,?)',
                         (row['id'], row.get('turn_id', 't-x'), row['hash'], row.get('subject', 'speaker'),
                          row.get('predicate', 'office_location'), row['value'], '{}', row.get('superseded_by'),
                          row.get('retracted_by')))
        for claim_id, message_hash in witnesses:
            conn.execute('INSERT INTO source_claim_witnesses(claim_id,turn_id,message_hash,data_json) '
                         'VALUES (?,?,?,?)', (claim_id, 't-x', message_hash, '{}'))
    if decided:
        conn = open_mind_db(home / 'mind.db')
        state = MindState(conn)
        for older, newer, verdict in decided:
            state.set(f'{RELATION_KEY}{_pair_id(older, newer)}', text=verdict, causes=[older, newer, 'note:n-1'])
        conn.close()
    return root


OLD = {'id': 'claim:old', 'hash': 'h-1', 'value': 'room 4'}
NEW = {'id': 'claim:new', 'hash': 'h-2', 'value': 'room 9'}


def test_an_old_value_the_store_replaced_is_a_supersession(storage, records, tmp_path):
    root = store(tmp_path / 's', [{**OLD, 'superseded_by': NEW['id']}, NEW])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.SUPERSEDED
    assert row['counts'] == {'evidence': 2, 'stored': 2, 'claims': 2, 'live': 1, 'superseded': 1, 'withdrawn': 0,
                             'replaces': 0, 'messages_live': 1, 'values_live': 1, 'latest_live': True}
    assert (row['question'], row['kind'], row['store']) == ('q-1', 'update', 'formed')


def test_a_value_withdrawn_is_a_supersession_too(storage, records, tmp_path):
    root = store(tmp_path / 's', [{**OLD, 'retracted_by': 'claim:retraction'}, NEW])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.SUPERSEDED and row['counts']['withdrawn'] == 1


@pytest.mark.parametrize('order', ['old-new', 'new-old'])
def test_two_live_values_the_night_decided_cannot_both_stand_are_an_unresolved_conflict(storage, records, tmp_path,
                                                                                        order):
    pair = (OLD['id'], NEW['id']) if order == 'old-new' else (NEW['id'], OLD['id'])
    root = store(tmp_path / 's', [OLD, NEW], decided=[(*pair, 'replaces')])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.CONFLICT and row['counts']['replaces'] == 1


def test_a_pair_decided_to_add_is_no_conflict(storage, records, tmp_path):
    root = store(tmp_path / 's', [OLD, NEW], decided=[(OLD['id'], NEW['id'], 'adds')])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.BOTH_CURRENT and row['counts']['replaces'] == 0


def test_live_values_from_two_evidence_messages_with_nothing_settling_them_are_both_current(storage, records,
                                                                                            tmp_path):
    """Whatever keys they are filed under: nothing in the store links them, and both read as current."""
    root = store(tmp_path / 's', [OLD, {**NEW, 'predicate': 'workplace'}])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.BOTH_CURRENT and row['counts']['messages_live'] == 2


def test_values_from_one_evidence_message_alone_are_a_single_value(storage, records, tmp_path):
    root = store(tmp_path / 's', [NEW])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.SINGLE_VALUE
    assert row['counts']['latest_live'] is True and row['counts']['claims'] == 1
    old_only = storage.outcome(store(tmp_path / 't', [OLD]), question(records))
    assert old_only['outcome'] == storage.SINGLE_VALUE and old_only['counts']['latest_live'] is False


def test_a_claim_witnessed_by_an_evidence_message_counts_as_its_claim(storage, records, tmp_path):
    """A restated value is a witness on the claim, not a new row: the claim belongs to the restating message too,
    and one value stated twice is still one value, never two held as current."""
    root = store(tmp_path / 's', [{**OLD, 'hash': 'h-3'}], witnesses=[(OLD['id'], 'h-1'), (OLD['id'], 'h-2')])
    row = storage.outcome(root, question(records))
    assert row['counts']['claims'] == 1 and row['counts']['messages_live'] == 2
    assert row['counts']['values_live'] == 1
    assert row['outcome'] == storage.SINGLE_VALUE


def test_one_claim_anchored_at_one_evidence_message_and_witnessed_by_the_other_is_a_single_value(storage, records,
                                                                                                 tmp_path):
    """The ledger anchors a restated claim at its newest witness: the witnesses are its provenance."""
    root = store(tmp_path / 's', [NEW], witnesses=[(NEW['id'], 'h-1'), (NEW['id'], 'h-2')])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.SINGLE_VALUE and row['counts']['latest_live'] is True


def test_the_same_value_filed_twice_from_two_evidence_messages_is_a_single_value(storage, records, tmp_path):
    root = store(tmp_path / 's', [{**OLD, 'value': 'room 9'}, {**NEW, 'predicate': 'workplace'}])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.SINGLE_VALUE and row['counts']['claims'] == 2
    assert (row['counts']['messages_live'], row['counts']['values_live']) == (2, 1)


def test_two_values_from_one_evidence_message_are_no_competing_values(storage, records, tmp_path):
    """Two things one message said are not an old and a new value."""
    root = store(tmp_path / 's', [NEW, {**NEW, 'id': 'claim:floor', 'predicate': 'office_floor', 'value': 'second'}])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.SINGLE_VALUE and row['counts']['values_live'] == 2


def test_a_restated_old_value_beside_a_new_one_is_both_current(storage, records, tmp_path):
    """The old value witnessed again by the latest message, and the new value there too: two values live."""
    root = store(tmp_path / 's', [OLD, NEW], witnesses=[(OLD['id'], 'h-2')])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.BOTH_CURRENT


def test_claims_from_other_messages_are_not_the_questions(storage, records, tmp_path):
    root = store(tmp_path / 's', [{**OLD, 'hash': 'h-3'}, {**NEW, 'hash': 'h-1a'}])
    row = storage.outcome(root, question(records))
    assert row['outcome'] == storage.NOT_STORED and row['counts']['claims'] == 0


def test_an_evidence_turn_the_store_never_held_is_counted_and_not_read(storage, records, tmp_path):
    root = store(tmp_path / 's', [NEW])
    row = storage.outcome(root, question(records, evidence=(('s-1', 0), ('s-2', 0), ('s-9', 4))))
    assert row['counts']['evidence'] == 3 and row['counts']['stored'] == 2


def test_the_import_only_store_holds_no_claim(storage, records, tmp_path):
    root = store(tmp_path / 's', [], kind='import_only')
    row = storage.outcome(root, question(records), store='import_only')
    assert row['outcome'] == storage.NOT_STORED and row['store'] == 'import_only'


def test_only_knowledge_updates_and_contradictions_have_a_storage_outcome(storage, records, tmp_path):
    root = store(tmp_path / 's', [OLD, NEW])
    assert storage.outcome(root, question(records, kind='contradiction_resolution'))['kind'] == 'conflict'
    for kind, abstention in (('temporal-reasoning', False), ('knowledge-update', True)):
        assert storage.outcome(root, question(records, kind=kind, abstention=abstention)) is None


def test_the_rows_cover_every_store_a_formed_directory_holds(storage, records, tmp_path):
    """A formed store and the copy taken before its night are two stores: each arm reads one."""
    root = store(tmp_path / 's', [OLD, NEW], decided=[(OLD['id'], NEW['id'], 'replaces')])
    store(root, [OLD, NEW], kind='before_night')
    output = tmp_path / 'storage.jsonl'
    stores = tmp_path / 'stores'
    stores.mkdir()
    root.rename(stores / storage.runner.safe_name('c-1'))       # where ``runner.py replay`` forms it
    counts = storage.read(stores, [question(records), question(records, kind='single-session-user', qid='q-2')],
                          output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert counts == {'rows': 2}
    assert {(row['question'], row['store'], row['outcome']) for row in rows} == {
        ('q-1', 'formed', storage.CONFLICT), ('q-1', 'before_night', storage.BOTH_CURRENT)}
    assert all(row['protocol'] == storage.PROTOCOL for row in rows)


def test_the_command_reads_stores_named_by_a_relative_directory(storage, records, tmp_path, monkeypatch, capsys):
    """``--stores stores``, as the page documents it: the store's databases are opened read-only by URI, which needs
    an absolute path."""
    root = store(tmp_path / 's', [OLD, NEW], decided=[(OLD['id'], NEW['id'], 'replaces')])
    (tmp_path / 'stores').mkdir()
    root.rename(tmp_path / 'stores' / storage.runner.safe_name('c-1'))
    data = records.Dataset('longmemeval_s', '0' * 64, {}, (question(records),), {})
    monkeypatch.setattr(storage.runner, 'load_dataset', lambda name, data_dir, offline: data)
    monkeypatch.chdir(tmp_path)
    assert storage.main(['--dataset', 'longmemeval_s', '--stores', 'stores', '--output', 'storage.jsonl']) == 0
    row, = [json.loads(line) for line in (tmp_path / 'storage.jsonl').read_text().splitlines()]
    assert (row['outcome'], row['counts']['claims'], row['counts']['replaces']) == (storage.CONFLICT, 2, 1)


def test_a_replayed_update_is_read_from_the_formed_store(storage, formed):
    """The replay's office moved from Room 4 to Room 9: before the night both are held as current, and the night
    decided the newer replaces the older, which the store keeps as a conflict for the owner to settle."""
    before = storage.outcome(formed.root, formed.question, store='before_night')
    after = storage.outcome(formed.root, formed.question)
    assert before['outcome'] == storage.BOTH_CURRENT
    assert after['outcome'] == storage.CONFLICT
    assert after['counts']['evidence'] == 2 and after['counts']['stored'] == 2
