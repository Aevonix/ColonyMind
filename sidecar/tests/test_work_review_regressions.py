"""Review regressions for optional work admission and terminal event delivery."""
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from test_work_projection import stores, admit, envelope, operation  # noqa: F401
from test_work_projection import event
from colonymind.work.verification import effect_kind, root_status, verify


def test_missing_parent_wait_expires(stores):
    admit(stores)
    body = event(envelope(parent_turn_id='unseen-child'))
    with pytest.raises(ValueError, match='work_parent_missing'):
        stores.work.admit_event(body, contact_id='owner', principal_id='host')
    with closing(stores.ledger._connect()) as conn, conn:
        conn.execute('UPDATE work_event_wait SET first_seen_at=?',
            ((datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat(),))
    with pytest.raises(ValueError, match='work_parent_expired'):
        stores.work.admit_event(body, contact_id='owner', principal_id='host')


def test_old_root_gives_new_missing_parent_event_a_fresh_wait(stores):
    admit(stores)
    with closing(stores.ledger._connect()) as conn, conn:
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn.execute('UPDATE turn_sources SET ingested_at=?', (old,))
        conn.execute('UPDATE work_envelopes SET created_at=?', (old,))
    body = event(envelope(parent_turn_id='unseen-child'))
    for _ in range(2):
        with pytest.raises(ValueError, match='work_parent_missing'):
            stores.work.admit_event(body, contact_id='owner', principal_id='host')
    with closing(stores.ledger._connect()) as conn:
        rows = conn.execute('SELECT first_seen_at FROM work_event_wait WHERE event_id=?', ('late',)).fetchall()
    assert len(rows) == 1
    assert datetime.now(timezone.utc) - datetime.fromisoformat(rows[0][0]) < timedelta(seconds=10)


def test_parent_admission_settles_wait_without_restarting_its_clock(stores):
    admit(stores)
    body = event(envelope(parent_turn_id='child'))
    with pytest.raises(ValueError, match='work_parent_missing'):
        stores.work.admit_event(body, contact_id='owner', principal_id='host')
    stores.work.admit_event(event(envelope(operations=[], children=[
        {'turn_id': 'child', 'parent_turn_id': 'root'}]), 'parent'), contact_id='owner', principal_id='host')
    assert stores.work.admit_event(body, contact_id='owner', principal_id='host')['accepted']
    with closing(stores.ledger._connect()) as conn:
        assert not conn.execute('SELECT 1 FROM work_event_wait WHERE event_id=?', ('late',)).fetchone()


def effect(value):
    return {'tool': value['tool'], 'resource_ref': value['resource_ref'],
            'kind': effect_kind(value), 'status': verify(value)[0]}


@pytest.mark.parametrize('tool,action', [('read_file', 'read'), ('terminal', 'probe')])
def test_failed_read_and_probe_do_not_fail_verified_work(tool, action):
    failed = operation('probe', tool=tool, action=action, success=False)
    assert root_status([effect(failed), effect(operation())]) == 'proven'


@pytest.mark.parametrize('replacement,expected', [
    ({}, 'proven'), ({'resource_ref': 'resource:other'}, 'failed'),
    ({'tool': 'patch'}, 'failed'), ({'evidence': {'kind': 'none'}}, 'failed'),
])
def test_only_later_verified_same_tool_same_resource_resolves_failure(replacement, expected):
    assert root_status([effect(operation('first', success=False)),
                        effect(operation('retry', **replacement))]) == expected


def test_arbitrary_failed_terminal_execution_is_still_failure():
    assert root_status([effect(operation(tool='terminal', action='execute', success=False)),
                        effect(operation('write'))]) == 'failed'


@pytest.mark.parametrize('value,kind,status', [
    (operation(tool='patch', action='delete', effect='remove'), 'remove', 'ran'),
    (operation(tool='kanban_create', action='dispatch', effect='dispatch'), 'dispatch', 'dispatched'),
    (operation(tool='custom_tool', action='', effect='write'), 'write', 'ran'),
])
def test_host_effect_is_retained_without_granting_an_unknown_tool_proof(value, kind, status):
    assert effect_kind(value) == kind
    assert verify(value)[0] == status


@pytest.mark.parametrize('action,expected', [
    ('create', 'create'), ('list', 'read'), ('update', 'edit'), ('pause', 'edit'),
    ('resume', 'edit'), ('remove', 'remove'), ('run', 'dispatch'), ('resnap', 'edit'),
])
def test_pinned_hermes_cron_action_classification(action, expected):
    assert effect_kind(operation(tool='cronjob', action=action)) == expected


def test_repeated_erased_owner_text_retains_other_source_messages(stores):
    stores.ledger.record_source('first', contact_id='owner', session_id='session',
        messages=[{'role': 'user', 'content': 'yes'}], derive_claims=False)
    from colonymind.memory import curated
    curated.forget_sources(stores.ledger, 'owner', source_ids=['first'])
    result = {}
    messages = [{'role': 'user', 'content': 'yes'}, {'role': 'assistant', 'content': 'Changed the file'}]
    assert stores.ledger.record_source('second', contact_id='owner', session_id='session', messages=messages,
        work={'envelope': envelope('second'), 'principal_id': 'host'}, work_result=result, derive_claims=False)
    assert result == {'recorded': False, 'reason': 'work_source_erased'}
    with closing(stores.ledger._connect()) as conn:
        assert conn.execute("SELECT 1 FROM turn_sources WHERE turn_id='second'").fetchone()
        assert not conn.execute("SELECT 1 FROM work_envelopes WHERE root_turn_id='second'").fetchone()
    assert not stores.ledger.record_source('second', contact_id='owner', session_id='session', messages=messages,
        work={'envelope': envelope('second'), 'principal_id': 'host'}, work_result=result, derive_claims=False)

def test_unadmitted_and_erased_roots_are_terminal(stores):
    body = event(envelope('guest'))
    with pytest.raises(ValueError, match='work_root_missing'):
        stores.work.admit_event(body, contact_id='owner', principal_id='host')
    stores.ledger.record_source('guest', contact_id='guest', session_id='session',
        messages=[{'role': 'user', 'content': 'A guest request'}], derive_claims=False)
    with pytest.raises(ValueError, match='work_root_ineligible'):
        stores.work.admit_event(body, contact_id='owner', principal_id='host')
    admit(stores)
    from colonymind.memory import curated
    curated.forget_sources(stores.ledger, 'owner', source_ids=['root'])
    with pytest.raises(ValueError, match='work_root_erased'):
        stores.work.admit_event(event(envelope()), contact_id='owner', principal_id='host')
