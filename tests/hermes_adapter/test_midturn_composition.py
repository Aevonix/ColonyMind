"""Native composed text keeps each admitted owner's capture and answer identity."""
import copy
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest



def test_rewrite_evidence_is_bound_to_the_observed_event():
    from dataclasses import FrozenInstanceError
    from colonymind_shared.midturn import begin_rewrite, bind_rewrite, matches_rewrite, seal_rewrite
    first, second = NS(text="original", message_id="same"), NS(text="original", message_id="same")
    receipt = begin_rewrite(first)
    heard = bind_rewrite(first)
    second._colonymind_text_receipt = receipt
    assert bind_rewrite(second) is None
    second._colonymind_text_receipt = copy.copy(receipt)
    assert bind_rewrite(second) is None
    with pytest.raises(FrozenInstanceError):
        receipt.before = 'changed'
    sealed = seal_rewrite(first, lambda _: "  sealed  ")
    first.text = sealed
    assert matches_rewrite(heard, sealed)
    assert matches_rewrite(heard, "sealed", strip=True)
    assert not matches_rewrite(copy.copy(heard), sealed)
    assert not matches_rewrite(heard, "sealed")
    assert not matches_rewrite(heard, "forged")
    seal_rewrite(first, lambda _: "second-transform")
    assert not matches_rewrite(heard, "second-transform")
    assert matches_rewrite(heard, sealed)


@pytest.mark.parametrize('change', ['source', 'sender', 'chat', 'message'])
def test_sealing_rejects_changed_source_coordinates(change):
    from colonymind_shared.midturn import begin_rewrite, seal_rewrite, matches_rewrite
    event = NS(text='original', source=NS(platform='sms', chat_id='chat', user_id='owner'), message_id='id')
    receipt = begin_rewrite(event)
    if change == 'source':
        event.source = copy.copy(event.source)
    elif change == 'message':
        event.message_id = 'other'
    else:
        setattr(event.source, 'user_id' if change == 'sender' else 'chat_id', 'other')
    seal_rewrite(event, lambda _: 'sealed')
    assert not matches_rewrite(receipt, 'sealed')


def test_receipt_is_consumed_once_per_ingress_observer():
    from colonymind_shared.midturn import bind_rewrite
    event = NS(text='original', message_id='id')
    observer, other = object(), object()
    assert bind_rewrite(event, observer=observer) is not None
    assert bind_rewrite(event, observer=observer) is None
    assert bind_rewrite(event, observer=other) is not None


def test_retained_bindings_verify_in_a_fresh_process_without_saving_words(tmp_path):
    from colonymind_shared.turn_bindings import TurnBindings
    path = tmp_path/'bindings.sqlite3'
    store = TurnBindings(path)
    scope = ('session', 'turn', 'sms', 'sender', 'chat', 'dm', 'last-id')
    members = (('first-id', 'Private ordinary candidate'), ('last-id', '[secret S-AAAAAAAA: password]'))
    text = 'Native composition envelope'
    evidence = [{'message_id': ident} for ident, _ in members]
    assert store.remember(scope, text, members, evidence, ())
    assert store.remember(scope, text, members, evidence, ())
    assert not store.remember(scope, text + ' addition', members, evidence, ())
    assert path.stat().st_mode & 0o777 == 0o600
    assert all(words.encode() not in path.read_bytes() for _, words in members)
    code = '''
import json, sys
from colonymind_shared.turn_bindings import TurnBindings
path, scope, text, members, evidence = json.loads(sys.argv[1])
store = TurnBindings(path)
assert store.recover(scope, text, members) == evidence
assert store.recover(scope, text + ' addition', members) is False
for index in range(len(scope)):
    other = list(scope)
    other[index] += '-changed'
    assert store.recover(other, text, members) is None
assert store.recover(scope, text, list(reversed(members))) is False
members[0][1] += ' new text'
assert store.recover(scope, text, members) is False
'''
    subprocess.run([sys.executable, '-B', '-c', code, json.dumps([str(path), scope, text, members, evidence])], check=True)


def test_composition_bindings_belong_to_the_instance(tmp_path):
    from colonymind_hermes.capture import SessionMap
    from hands_fake_hermes import FakeContext, load_hands
    from importlib import import_module
    from colonymind_shared.turn_bindings import TurnBindings
    shared = tmp_path/'shared-hermes'
    maps, guards = [], []
    for name in ['first', 'second']:
        home = tmp_path/name
        settings = NS(home=home, outbox_path=shared/'state/capture.sqlite3', hermes_home=shared)
        maps.append(SessionMap(settings, NS()))
        runtime = NS(config={}, client=NS(settings=settings), spool_dir=home/'hands-spool')
        front = import_module(load_hands().__name__ + '.front')
        guards.append(front.FrontGuard(runtime, FakeContext(), start_threads=False))
    try:
        for stores in [(m._composed_bindings for m in maps), (g._composed_bindings for g in guards)]:
            first, second = stores
            scope = ('session', 'turn', 'sms', 'sender', 'chat', 'last-id')
            assert first.path != second.path
            assert first.remember(scope, 'envelope', (('last-id', 'words'),), [{'message_id': 'last-id'}], ())
            assert second.recover(scope, 'envelope', (('last-id', 'words'),)) is None
            assert TurnBindings(first.path).recover(scope, 'envelope', (('last-id', 'words'),)) is not None
    finally:
        for guard in guards:
            guard.close()


def test_receipt_retains_no_original_plaintext():
    from colonymind_shared.midturn import begin_rewrite, seal_rewrite
    event = NS(text='synthetic-sensitive-input', message_id='id')
    receipt = begin_rewrite(event)
    event.text = seal_rewrite(event, lambda _: 'sealed')
    assert 'synthetic-sensitive-input' not in repr(vars(receipt))


def test_receipt_does_not_keep_original_event_alive():
    import gc
    import weakref
    from colonymind_shared.midturn import begin_rewrite, bind_rewrite
    class Event:
        pass
    event = Event()
    event.text, event.message_id, event.source = 'original', 'id', NS(platform='sms')
    receipt = begin_rewrite(event)
    event_ref = weakref.ref(event)
    del event
    gc.collect()
    assert event_ref() is None
    assert receipt.identity() is None
    replacement = Event()
    replacement.text, replacement.message_id, replacement.source = 'original', 'id', NS(platform='sms')
    replacement._colonymind_text_receipt = receipt
    assert bind_rewrite(replacement) is None
