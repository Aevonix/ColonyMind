"""Recurring-delivery reparenting: scope, proof by readback, and honest refusal.

The engine is driven exactly as ``ArchiveStore`` drives it, through the five
phases, with a fake standing in for the public ``cron.jobs`` API. Every mutation
here is expected to be proved by an independent read, so the fake records its
calls and can lie the way a real scheduler can: persist and then raise, change
underneath us, or refuse to read back.
"""
from contextlib import closing
import importlib.util
import json
import sqlite3

import pytest

from colonymind.session_archive import reparent as engine_module
from colonymind.session_archive.reparent import archive_reparent
from colonymind.session_archive.store import ArchiveError, ArchiveStore

needs_hermes = pytest.mark.skipif(importlib.util.find_spec("hermes_time") is None,
                                  reason="needs stock Hermes in the test interpreter")
pytestmark = needs_hermes

SOURCE = {'platform': 'whatsapp', 'chat_id': 'chat-side', 'thread_id': None}
MAIN = {'platform': 'whatsapp', 'chat_id': 'chat-main', 'thread_id': None}


def job(ident='job-1', origin=SOURCE, deliver='origin', **changes):
    row = {'id': ident, 'name': ident, 'schedule': {'kind': 'interval'}, 'deliver': deliver,
           'origin': None if origin is None else dict(origin), 'enabled': True,
           'attach_to_session': True, 'prompt': 'report', **changes}
    return row


class Cron:
    """A fake public cron API: ids to job dicts, with recorded calls."""

    def __init__(self, *jobs, include_disabled_required=True):
        self.jobs = {row['id']: dict(row) for row in jobs}
        self.updates = []
        self.reads = []
        self.listed = []
        self.include_disabled_required = include_disabled_required
        self.persist_then_raise = set()
        self.refuse_update = set()
        self.unreadable = set()
        self.drop_on_read = set()

    def list_jobs(self, include_disabled=False):
        self.listed.append(include_disabled)
        if self.include_disabled_required and not include_disabled:
            raise AssertionError('a paused recurring delivery must be listed too')
        return [dict(row) for row in self.jobs.values()]

    def get_job(self, job_id):
        self.reads.append(job_id)
        if job_id in self.unreadable:
            raise OSError('scheduler unavailable')
        if job_id in self.drop_on_read:
            return None
        row = self.jobs.get(job_id)
        return dict(row) if row else None

    def update_job(self, job_id, updates):
        self.updates.append((job_id, updates))
        if job_id in self.refuse_update:
            raise RuntimeError('update refused')
        self.jobs[job_id].update(updates)
        if job_id in self.persist_then_raise:
            raise RuntimeError('wrote the change, then failed')
        return None             # Never trusted; a real update returns the record.


@pytest.fixture
def cron(monkeypatch):
    def install(*jobs, **kwargs):
        fake = Cron(*jobs, **kwargs)
        monkeypatch.setattr(engine_module, '_core',
                            lambda: (fake.list_jobs, fake.get_job, fake.update_job))
        return fake
    return install


def request(reparent=True, source=SOURCE, main=MAIN, **changes):
    return {'archived': True, 'reparent_to_main': reparent, 'operation_id': 'op',
            'main_session_id': 'session-main',
            'source_origin': source if source is None else dict(source),
            'main_origin': main if main is None else dict(main), **changes}


def run(phase, *, session='side', body=None, intent=None, connection=None):
    return archive_reparent(phase=phase, session_id=session,
                            request=request() if body is None else body,
                            intent=intent, connection=connection)


def plan(body=None):
    return run('plan', body=body)


# ── scope ──────────────────────────────────────────────────────────────────────

def test_only_an_exactly_matching_origin_is_in_scope(cron):
    other_chat = {'platform': 'whatsapp', 'chat_id': 'chat-other', 'thread_id': None}
    other_platform = {'platform': 'sms', 'chat_id': 'chat-side', 'thread_id': None}
    cron(job('in'), job('other-chat', other_chat), job('other-platform', other_platform),
         job('no-origin', None), {**job('string-origin'), 'origin': 'digest-replaces-x'},
         job('partial', {'platform': 'whatsapp', 'chat_id': ''}))
    assert [row['id'] for row in plan()['jobs']] == ['in']


def test_a_thread_pinned_job_is_not_moved_by_archiving_the_chat(cron):
    # Core treats thread_id as a per-delivery detail it may drop when stale, but a
    # thread is its own conversation here: archiving the chat must not silently
    # retarget work the owner pinned to a thread inside it.
    pinned = {'platform': 'whatsapp', 'chat_id': 'chat-side', 'thread_id': 't-9'}
    cron(job('plain'), job('pinned', pinned))
    assert [row['id'] for row in plan()['jobs']] == ['plain']


def test_platform_case_does_not_decide_identity(cron):
    cron(job('mixed', {'platform': 'WhatsApp', 'chat_id': 'chat-side', 'thread_id': None}))
    assert [row['id'] for row in plan()['jobs']] == ['mixed']


@pytest.mark.parametrize('lanes, moved', [
    ({'deliver': 'origin'}, True),
    ({'deliver': 'origin,local'}, True),
    ({'deliver': 'local,origin'}, True),
    ({'deliver': ' origin , whatsapp:other '}, True),
    ({'deliver': ['local', 'origin']}, True),
    ({'deliver': 'local', 'failure_deliver': 'origin,local'}, True),
    ({'deliver': 'local', 'failure_deliver': 'origin'}, True),
    ({'deliver': 'ORIGIN'}, False),
    ({'deliver': 'Origin'}, False),
    ({'deliver': 'originate'}, False),
    ({'deliver': 'all'}, False),
    ({'deliver': 'local'}, False),
    ({'deliver': 'whatsapp:chat-side'}, False),
    ({'deliver': 'local', 'failure_deliver': 'local'}, False),
    ({'deliver': None}, False),
    ({'deliver': []}, False),
])
def test_only_a_job_that_actually_delivers_there_is_moved(cron, lanes, moved):
    # A job that merely records the chat as provenance delivers nowhere near it.
    # Retargeting it would move no delivery and would rewrite exactly the
    # provenance this capability refuses to rewrite.
    cron(job('candidate', **lanes))
    assert bool(plan()['jobs']) is moved


def test_paused_and_disabled_recurring_deliveries_are_moved_too(cron):
    fake = cron(job('paused', paused=True), job('disabled', enabled=False))
    assert [row['id'] for row in plan()['jobs']] == ['disabled', 'paused']
    assert fake.listed == [True]


def test_the_plan_records_both_origins_and_sorts_by_id(cron):
    cron(job('b'), job('a'))
    planned = plan()['jobs']
    assert [row['id'] for row in planned] == ['a', 'b']
    assert planned[0] == {'id': 'a', 'original_origin': dict(SOURCE),
                          'intended_origin': dict(MAIN)}


def test_an_unusable_job_inventory_is_refused_rather_than_guessed(cron):
    cron(job('ok'), job('', origin=SOURCE))
    with pytest.raises(ArchiveError) as refused:
        plan()
    assert refused.value.code == 'source_unavailable'


def test_declining_reparenting_reports_the_work_it_did_not_move(cron):
    cron(job('in'))
    body = request(reparent=False)
    planned = plan(body)
    assert planned == {'jobs': [], 'declined': ['in']}
    assert run('apply', body=body, intent=planned) == {'reparented': {'cron_jobs': 0},
                                                       'tasks_reparented': False}


def test_declining_with_nothing_in_scope_is_honestly_true(cron):
    cron(job('elsewhere', {'platform': 'sms', 'chat_id': 'x', 'thread_id': None}))
    body = request(reparent=False)
    planned = plan(body)
    assert planned == {'jobs': [], 'declined': []}
    assert run('apply', body=body, intent=planned)['tasks_reparented'] is True


# ── apply ──────────────────────────────────────────────────────────────────────

def test_apply_sets_the_complete_origin_and_changes_nothing_else(cron):
    pinned = {'platform': 'whatsapp', 'chat_id': 'chat-side', 'thread_id': None}
    fake = cron(job('one', pinned, workdir='/tmp/x'))
    before = dict(fake.jobs['one'])
    planned = plan()
    assert run('apply', intent=planned) == {'reparented': {'cron_jobs': 1},
                                            'tasks_reparented': True}
    assert fake.updates == [('one', {'origin': dict(MAIN)})]
    after = fake.jobs['one']
    assert after['origin'] == dict(MAIN)          # a SET, so no thread_id survives
    assert {k: v for k, v in after.items() if k != 'origin'} == \
           {k: v for k, v in before.items() if k != 'origin'}


def test_apply_proves_every_move_by_reading_the_job_back(cron):
    fake = cron(job('one'))
    planned = plan()
    run('apply', intent=planned)
    # Before the set and after the set. Origin and delivery lane come from ONE read
    # each time: reading them separately let a job change origin between the two and
    # pass both checks. The update's return value is never trusted. The final proof
    # is the re-listing, not a third per-job read, so that the moved jobs and the
    # source inventory are judged from the same snapshot.
    assert fake.reads == ['one'] * 2
    assert fake.listed[-1] is True


def test_a_job_this_operation_already_moved_is_not_updated_again(cron):
    # Already at Main AND durably flagged as attempted by this same operation: our
    # own earlier attempt, so it is not updated again and it stays ours to restore.
    fake = cron(job('one', MAIN))
    intent = {'jobs': [{'id': 'one', 'original_origin': dict(SOURCE),
                        'intended_origin': dict(MAIN), 'attempted': True}], 'declined': []}
    assert run('apply', intent=intent)['tasks_reparented'] is True
    assert fake.updates == []
    assert intent['jobs'][0]['moved'] is True


def test_a_job_someone_else_put_at_main_is_not_claimed_as_ours(cron):
    # Same end state, no attempt of ours behind it: another caller moved it there.
    # Claiming it would make a later rollback drag their work back to the archived
    # chat, which is exactly what review round two reproduced.
    fake = cron(job('one', MAIN))
    intent = {'jobs': [{'id': 'one', 'original_origin': dict(SOURCE),
                        'intended_origin': dict(MAIN)}], 'declined': []}
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=intent)
    assert refused.value.code == 'uncertain'
    assert refused.value.details['conflicted'] == ['one']
    assert refused.value.details['reason'] == 'job_moved_by_another'
    assert fake.updates == []
    assert not intent['jobs'][0].get('moved')


def test_a_job_moved_by_someone_else_is_refused_not_overwritten(cron):
    elsewhere = {'platform': 'sms', 'chat_id': 'chat-third', 'thread_id': None}
    fake = cron(job('one', elsewhere))
    intent = {'jobs': [{'id': 'one', 'original_origin': dict(SOURCE),
                        'intended_origin': dict(MAIN)}], 'declined': []}
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=intent)
    assert refused.value.code == 'uncertain'
    assert fake.updates == [] and fake.jobs['one']['origin'] == elsewhere


def test_an_unverifiable_move_is_uncertain_even_though_the_update_returned(cron):
    fake = cron(job('one'))
    fake.update_job = lambda job_id, updates: fake.updates.append((job_id, updates))
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=plan())
    assert refused.value.code == 'uncertain'


def test_a_vanishing_job_is_a_source_failure_not_a_silent_skip(cron):
    fake = cron(job('one'))
    fake.drop_on_read.add('one')
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent={'jobs': [{'id': 'one', 'original_origin': dict(SOURCE),
                                       'intended_origin': dict(MAIN)}], 'declined': []})
    assert refused.value.code == 'source_unavailable'


# ── rollback ───────────────────────────────────────────────────────────────────

def moved_intent(*ids, attempted=True, moved=True):
    return {'jobs': [{'id': ident, 'original_origin': dict(SOURCE),
                      'intended_origin': dict(MAIN), 'attempted': attempted,
                      'moved': moved}
                     for ident in ids], 'declined': []}


def test_rollback_restores_and_verifies_every_origin(cron):
    fake = cron(job('one', MAIN), job('two', MAIN))
    assert run('rollback', intent=moved_intent('one', 'two')) == {
        'restored': {'cron_jobs': 2}, 'half_applied': False, 'jobs_left_moved': []}
    assert all(row['origin'] == dict(SOURCE) for row in fake.jobs.values())


def test_rollback_is_idempotent_on_a_job_already_back(cron):
    fake = cron(job('one', SOURCE))
    assert run('rollback', intent=moved_intent('one'))['restored'] == {'cron_jobs': 1}
    assert fake.updates == []


def test_a_restore_that_persisted_then_raised_is_still_verified(cron):
    fake = cron(job('one', MAIN))
    fake.persist_then_raise.add('one')
    assert run('rollback', intent=moved_intent('one')) == {
        'restored': {'cron_jobs': 1}, 'half_applied': False, 'jobs_left_moved': []}


def test_an_unverifiable_restore_names_the_job_left_moved(cron):
    fake = cron(job('one', MAIN), job('two', MAIN))
    fake.refuse_update.add('two')
    assert run('rollback', intent=moved_intent('one', 'two')) == {
        'restored': {'cron_jobs': 1}, 'half_applied': True, 'jobs_left_moved': ['two']}


def test_an_unreadable_job_is_named_and_never_written_blind(cron):
    # A job we cannot READ is left exactly as it is. Writing SOURCE without first
    # establishing where the job currently sits is how an independent move to a
    # third destination got overwritten after one transient read failure.
    fake = cron(job('one', MAIN))
    fake.unreadable.add('one')
    result = run('rollback', intent=moved_intent('one'))
    assert fake.updates == []
    assert result['jobs_left_moved'] == ['one'] and result['half_applied'] is True


# ── validate and commit ────────────────────────────────────────────────────────

def test_validate_tolerates_the_empty_request_of_a_receipt_read(cron):
    cron()
    assert run('validate', body={}) is None


@pytest.mark.parametrize('body', [
    request(main=dict(SOURCE)),
    request(main={'platform': 'WhatsApp', 'chat_id': 'chat-side', 'thread_id': None}),
    request(source={'platform': 'whatsapp'}),
    request(main=None),
])
def test_validate_refuses_a_no_op_or_malformed_pair(cron, body):
    cron()
    with pytest.raises(ArchiveError) as refused:
        run('validate', body=body)
    assert refused.value.code == 'invalid_request'


def test_commit_stages_nothing_because_nothing_local_moves(cron, tmp_path):
    cron()
    with closing(sqlite3.connect(tmp_path / 'overlay.db')) as conn:
        assert run('commit', intent=moved_intent('one'), connection=conn) is None
        attached = [row[1] for row in conn.execute('PRAGMA database_list')]
        assert attached == ['main']


def test_an_unknown_phase_is_refused(cron):
    cron()
    with pytest.raises(ArchiveError):
        run('sideways')


# ── end to end through the real overlay store ──────────────────────────────────

def test_the_store_and_engine_together_move_delivery_and_issue_one_receipt(cron, tmp_path):
    fake = cron(job('due'), job('provenance-only', deliver='local'),
                job('elsewhere', {'platform': 'sms', 'chat_id': 'x', 'thread_id': None}))
    store = ArchiveStore(tmp_path / 'archive.db')
    receipt = store.archive('side', request(), archive_reparent)
    assert receipt['archived'] is True and receipt['tasks_reparented'] is True
    assert receipt['reparented'] == {'cron_jobs': 1}
    assert fake.jobs['due']['origin'] == dict(MAIN)
    assert fake.jobs['provenance-only']['origin'] == dict(SOURCE)
    assert fake.jobs['elsewhere']['origin'] == {'platform': 'sms', 'chat_id': 'x', 'thread_id': None}
    # Replay is the same receipt and moves nothing a second time.
    assert store.archive('side', request(), archive_reparent) == receipt
    assert len(fake.updates) == 1


def test_a_failed_move_refuses_the_archive_and_puts_the_job_back(cron, tmp_path):
    fake = cron(job('due'))
    fake.update_job = lambda job_id, updates: fake.updates.append((job_id, updates))
    store = ArchiveStore(tmp_path / 'archive.db')
    with pytest.raises(ArchiveError) as refused:
        store.archive('side', request(), archive_reparent)
    assert refused.value.code == 'uncertain'
    assert refused.value.details['tasks_reparented'] is False
    assert fake.jobs['due']['origin'] == dict(SOURCE)
    assert store.receipt('side', archive_reparent)['receipt'] is None


def test_the_durable_operation_keeps_every_before_and_after_origin(cron, tmp_path):
    # A wrong source_origin from the caller has to be mechanically reversible, so
    # the completed operation row keeps each job's original and intended origin.
    cron(job('due'))
    store = ArchiveStore(tmp_path / 'archive.db')
    store.archive('side', request(), archive_reparent)
    with closing(sqlite3.connect(store.path)) as db:
        state, intent = db.execute(
            'SELECT state, intent_json FROM archive_operations').fetchone()
    assert state == 'completed'
    assert json.loads(intent)['jobs'] == [
        {'id': 'due', 'original_origin': dict(SOURCE), 'intended_origin': dict(MAIN),
         'attempted': True, 'moved': True}]


def test_an_archive_survives_a_work_root_rebuild(cron, tmp_path, monkeypatch):
    """The production failure this design avoids, pinned so it cannot come back.

    ``work_roots`` is a projection rebuilt from each root envelope, so anything
    resting on a write to ``work_roots.session_id`` would report success and then
    be silently undone by the next rebuild. Reparenting therefore moves recurring
    delivery only. This proves the two halves do not interact: a real rebuild
    leaves the archive overlay and every retargeted cron origin exactly as they
    were, and leaves the work root attributed to the session it ran in.
    """
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.work.store import WorkStore
    from test_work_projection import envelope

    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_URL', '')
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    work = WorkStore(ledger)
    ledger.record_source('root', contact_id='owner', session_id='side',
        messages=[{'role': 'user', 'content': 'Please change the report.'}],
        work={'envelope': envelope('root'), 'principal_id': 'host'}, derive_claims=False)
    work.wait_pending(timeout=5)
    work.reconcile()

    def roots():
        with closing(work._connect()) as conn:
            return [tuple(row) for row in conn.execute('SELECT id, session_id FROM work_roots')]

    fake = cron(job('due'))
    store = ArchiveStore(tmp_path / 'archive.db')
    receipt = store.archive('side', request(), archive_reparent)
    assert receipt['tasks_reparented'] is True
    assert roots() == [('root', 'side')]          # history stays where it happened

    work.rebuild_root('root')

    assert roots() == [('root', 'side')]
    assert fake.jobs['due']['origin'] == dict(MAIN)
    assert store.receipt('side', archive_reparent)['receipt'] == receipt
    assert store.archive('side', request(), archive_reparent) == receipt


# ── the review's findings, each pinned ─────────────────────────────────────────

def test_rollback_leaves_a_job_this_operation_never_touched(cron):
    """The P1 that would have redirected someone else's work.

    Apply refuses when it finds a planned job somewhere unexpected. Restoring the
    whole plan then dragged that job back from the third destination its own
    caller had chosen, and reported a clean refusal with no job ids.
    """
    third = {'platform': 'sms', 'chat_id': 'chat-third', 'thread_id': None}
    fake = cron(job('a', MAIN), job('b', third))
    intent = {'jobs': [{'id': 'a', 'original_origin': dict(SOURCE),
                        'intended_origin': dict(MAIN), 'attempted': True},
                       {'id': 'b', 'original_origin': dict(SOURCE),
                        'intended_origin': dict(MAIN), 'attempted': False}],
              'declined': []}
    result = run('rollback', intent=intent)
    assert fake.jobs['a']['origin'] == dict(SOURCE)
    assert fake.jobs['b']['origin'] == third          # untouched
    assert result == {'restored': {'cron_jobs': 1}, 'half_applied': False,
                      'jobs_left_moved': []}
    assert ('b', {'origin': dict(SOURCE)}) not in fake.updates


def test_rollback_names_a_job_that_moved_again_after_we_moved_it(cron):
    third = {'platform': 'sms', 'chat_id': 'chat-third', 'thread_id': None}
    fake = cron(job('a', third))
    result = run('rollback', intent=moved_intent('a'))
    assert fake.jobs['a']['origin'] == third          # not overwritten
    assert result['jobs_moved_elsewhere'] == ['a']
    assert result['restored'] == {'cron_jobs': 0} and result['half_applied'] is False


def test_a_job_that_drifts_back_before_the_commit_is_not_reported_as_moved(cron):
    """The P1 a per-job readback cannot catch: one job returns while another moves."""
    fake = cron(job('a'), job('b'))
    planned = plan()
    original = fake.update_job

    def update(job_id, updates):
        original(job_id, updates)
        if job_id == 'b':                 # a goes home while b is being moved
            fake.jobs['a']['origin'] = dict(SOURCE)
    fake.update_job = update
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=planned)
    assert refused.value.code == 'uncertain'
    assert refused.value.details['unverified'] == ['a']


def test_a_lane_that_stops_delivering_before_the_commit_is_refused(cron):
    fake = cron(job('one'))
    original = fake.update_job

    def update(job_id, updates):
        original(job_id, updates)
        fake.jobs[job_id]['deliver'] = 'local'   # now provenance only
    fake.update_job = update
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=plan())
    assert refused.value.details['unverified'] == ['one']


def test_a_core_refusal_names_the_job_instead_of_failing_silently(cron):
    """Core raises for an enabled one-shot whose time has passed. Only in-scope
    jobs reach update_job, and the owner is told which one refused."""
    fake = cron(job('stale-oneshot'))
    fake.refuse_update.add('stale-oneshot')
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=plan())
    assert refused.value.code == 'uncertain'
    assert refused.value.details == {'refused': ['stale-oneshot'],
                                     'reason': 'cron_update_refused'}


def test_an_interrupted_operation_is_not_read_as_clean(cron, tmp_path):
    fake = cron(job('due'))
    store = ArchiveStore(tmp_path / 'archive.db')
    original = fake.update_job

    def update(job_id, updates):
        original(job_id, updates)
        raise SystemExit('killed after the move persisted')
    fake.update_job = update
    with pytest.raises(SystemExit):
        store.archive('side', request(), archive_reparent)
    state = store.receipt('side', archive_reparent)
    assert state['half_applied'] is True and state['jobs_left_moved'] == ['due']
    assert state['receipt'] is None


def test_a_refusal_does_not_misreport_an_already_archived_session(cron, tmp_path):
    fake = cron(job('due'))
    store = ArchiveStore(tmp_path / 'archive.db')
    store.archive('side', request(), archive_reparent)
    # Now attempt to unarchive while core refuses the move back.
    fake.jobs['later'] = job('later', SOURCE)
    fake.refuse_update.add('later')
    body = dict(request(), archived=False, operation_id='unarchive')
    with pytest.raises(ArchiveError) as refused:
        store.archive('side', body, archive_reparent)
    assert refused.value.details['archived'] is True       # the overlay is unchanged
    assert store.receipt('side', archive_reparent)['archived'] is True


# ── classification is core's answer, not ours (review round two, P1 #1) ────────

@pytest.mark.parametrize('lanes, moved', [
    # A bare platform token equal to the origin's platform falls back to the ORIGIN
    # chat when that platform has no configured home channel, so it follows the
    # origin exactly like the ``origin`` token. Our own token rule missed this and
    # reported tasks_reparented true with cron_jobs 0 while the job kept firing
    # into the archived chat.
    ({'deliver': 'whatsapp'}, True),
    ({'deliver': 'local,whatsapp'}, True),
    ({'deliver': 'local', 'failure_deliver': 'whatsapp'}, True),
    # An EXPLICIT platform:chat target resolves to the same chat whatever the
    # origin says, so moving the origin would redirect nothing. Claiming it moved
    # would be the same lie in the other direction.
    ({'deliver': 'whatsapp:chat-side'}, False),
    ({'deliver': 'origin,whatsapp:chat-side'}, False),
])
def test_scope_follows_cores_own_resolver_not_our_reading_of_it(cron, lanes, moved):
    cron(job('candidate', **lanes))
    assert bool(plan()['jobs']) is moved


def test_a_bare_platform_job_is_actually_retargeted(cron):
    fake = cron(job('bare', deliver='whatsapp'))
    assert run('apply', intent=plan()) == {'reparented': {'cron_jobs': 1},
                                           'tasks_reparented': True}
    assert fake.jobs['bare']['origin'] == dict(MAIN)


def test_a_missing_core_resolver_refuses_loudly_instead_of_guessing(cron, monkeypatch):
    # No fallback to a rule of our own: a refused archive costs the owner a retry,
    # a false receipt costs them a reminder that still fires into a chat they
    # archived. Two review rounds found two branches our own rule got wrong.
    cron(job('one'))
    import cron.scheduler_delivery as delivery
    monkeypatch.delattr(delivery, '_resolve_delivery_targets')
    with pytest.raises(ArchiveError) as refused:
        plan()
    assert refused.value.code == 'uncertain'
    assert refused.value.details['reason'] == 'core_resolver_unavailable'


def test_a_resolver_that_raises_is_also_a_refusal(cron, monkeypatch):
    cron(job('one'))
    import cron.scheduler_delivery as delivery

    def explode(*_args, **_kwargs):
        raise RuntimeError('core said no')

    monkeypatch.setattr(delivery, '_resolve_delivery_targets', explode)
    with pytest.raises(ArchiveError) as refused:
        plan()
    assert refused.value.details['reason'] == 'core_resolver_failed'


def test_a_move_that_would_lose_the_delivery_is_refused_by_name(cron, monkeypatch):
    # The origin decides this job's delivery, but with Main as the origin core
    # resolves no target at all: moving it would silently drop a delivery the
    # owner still expects. Named at plan time rather than mutated and rolled back.
    cron(job('fragile'))
    import cron.scheduler_delivery as delivery
    real = delivery._resolve_delivery_targets

    def resolve(job_dict, *, for_failure=False):
        if (job_dict.get('origin') or {}).get('chat_id') == MAIN['chat_id']:
            return []
        return real(job_dict, for_failure=for_failure)

    monkeypatch.setattr(delivery, '_resolve_delivery_targets', resolve)
    with pytest.raises(ArchiveError) as refused:
        plan()
    assert refused.value.details == {'refused': ['fragile'],
                                     'reason': 'delivery_would_be_lost'}


# ── the pre-commit scan covers the whole set (review round two, P1 #2) ─────────

def test_a_job_that_becomes_in_scope_during_apply_is_not_left_behind(cron):
    # The receipt promises nothing is still aimed at the archived chat. A job
    # created while we worked was never in the planned set, so re-reading that set
    # could not see it: the scan has to re-list.
    fake = cron(job('planned'))
    planned = plan()

    original_update = fake.update_job

    def update_then_add(job_id, updates):
        result = original_update(job_id, updates)
        fake.jobs['late'] = job('late')
        return result

    fake.update_job = update_then_add
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=planned)
    assert refused.value.details['unverified'] == ['late']


def test_a_declined_archive_does_not_refuse_over_the_jobs_it_left(cron):
    # The whole-set scan belongs to a REPARENTING archive. A declined one promises
    # the opposite, so a job still aimed at the chat is the expected outcome.
    cron(job('left-alone'))
    body = request(reparent=False)
    assert run('apply', body=body, intent=plan(body)) == {
        'reparented': {'cron_jobs': 0}, 'tasks_reparented': False}


def test_a_job_that_drifts_back_during_the_scan_is_caught_by_one_read(cron):
    fake = cron(job('one'))
    planned = plan()
    reads = {'n': 0}
    original_get = fake.get_job

    def get_then_drift(job_id):
        reads['n'] += 1
        row = original_get(job_id)
        if reads['n'] == 2:         # the post-update readback has just passed
            fake.jobs[job_id]['origin'] = dict(SOURCE)
        return row

    fake.get_job = get_then_drift
    with pytest.raises(ArchiveError) as refused:
        run('apply', intent=planned)
    assert refused.value.details['unverified'] == ['one']


# ── rollback compensates only what this operation moved (P1 #3) ────────────────

def test_rollback_skips_a_job_this_operation_never_moved(cron):
    fake = cron(job('ours', MAIN), job('theirs', MAIN))
    intent = moved_intent('ours')
    intent['jobs'].append({'id': 'theirs', 'original_origin': dict(SOURCE),
                           'intended_origin': dict(MAIN)})
    assert run('rollback', intent=intent) == {
        'restored': {'cron_jobs': 1}, 'half_applied': False, 'jobs_left_moved': []}
    assert fake.jobs['theirs']['origin'] == dict(MAIN)
    assert [job_id for job_id, _ in fake.updates] == ['ours']


def test_rollback_names_a_job_that_moved_on_after_we_moved_it(cron):
    third = {'platform': 'sms', 'chat_id': 'chat-third', 'thread_id': None}
    fake = cron(job('one', third))
    result = run('rollback', intent=moved_intent('one'))
    assert result['jobs_moved_elsewhere'] == ['one']
    assert result['jobs_left_moved'] == [] and result['half_applied'] is False
    assert fake.updates == []


# ── bookkeeping is never why a job stays moved (review round two, P1 #4, P2 #6) ─

def test_a_core_refusal_reaches_the_owner_through_the_store(cron, tmp_path):
    # The engine names the job core refused and why. The rollback result must not
    # replace that: it is the only part of the answer the owner can act on.
    fake = cron(job('stale'))
    fake.refuse_update.add('stale')
    store = ArchiveStore(tmp_path / 'archive.db')
    with pytest.raises(ArchiveError) as refused:
        store.archive('side', request(), archive_reparent)
    assert refused.value.details['refused'] == ['stale']
    assert refused.value.details['reason'] == 'cron_update_refused'
    assert refused.value.details['half_applied'] is False
    assert fake.jobs['stale']['origin'] == dict(SOURCE)


class FailingOn:
    """A connection proxy that fails every statement mentioning one word."""

    def __init__(self, db, word):
        self.db, self.word, self.failed = db, word, 0

    def execute(self, sql, *args):
        if self.word in sql:
            self.failed += 1
            raise sqlite3.OperationalError('refusing %s' % self.word)
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_compensation_runs_when_our_own_operation_row_cannot_be_read(cron, tmp_path):
    cron(job('one'))
    store = ArchiveStore(tmp_path / 'archive.db')
    fallback = {'operation_id': 'op', 'destination_id': 'side',
                'request_json': '{}', 'intent_json': '{}'}

    class Dead:
        def execute(self, *_args):
            raise sqlite3.OperationalError('no')

    assert store._operation_row(Dead(), 'op', fallback) is fallback


def test_a_failed_refusal_write_still_reports_the_compensation_it_did(cron, tmp_path):
    # Compensation has already run and been verified; all that failed is writing it
    # down. Discarding the result left the job at Main with the owner told only
    # "source_unavailable", and told nothing about the job.
    fake = cron(job('one', MAIN))
    store = ArchiveStore(tmp_path / 'archive.db')
    with closing(sqlite3.connect(store.path)) as setup:
        setup.row_factory = sqlite3.Row
        setup.executescript('''
            CREATE TABLE IF NOT EXISTS archive_sessions (
                destination_id TEXT PRIMARY KEY, archived INTEGER NOT NULL,
                revision TEXT NOT NULL, archived_at TEXT, receipt_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS archive_operations (
                operation_id TEXT PRIMARY KEY, destination_id TEXT NOT NULL,
                request_json TEXT NOT NULL, intent_json TEXT NOT NULL,
                state TEXT NOT NULL, refusal_json TEXT, receipt_json TEXT);''')
        setup.execute('INSERT INTO archive_operations VALUES (?,?,?,?,?,?,?)',
                      ('op', 'side', json.dumps(request()),
                       json.dumps(moved_intent('one')), 'applying', None, None))
        setup.commit()
        row = setup.execute('SELECT * FROM archive_operations').fetchone()
        refusal = store._rollback(FailingOn(setup, 'refusal_json'), archive_reparent, row)
    assert refusal['restored'] == {'cron_jobs': 1}
    assert refusal['half_applied'] is False
    assert refusal['receipt_incomplete'] is True
    assert fake.jobs['one']['origin'] == dict(SOURCE)


# ── reviewer reproductions, pinned ─────────────────────────────────────────────

def test_a_lane_that_turns_back_to_the_source_after_the_move_is_not_a_clean_receipt(cron):
    """Third-round P1: the final proof read the moved jobs separately from the
    source inventory, so a job whose lane changed to an explicit SOURCE target
    after we moved it passed both checks - the per-job read only compared origins,
    and the inventory filter skipped the row because it no longer ORIGINATED at the
    source. One snapshot answers both questions."""
    fake = cron(job('one'))
    planned = plan()
    real_get = fake.get_job

    def turn_the_lane(job_id):
        row = real_get(job_id)
        if row and row.get('origin') == MAIN:
            fake.jobs[job_id]['deliver'] = 'whatsapp:chat-side'
        return row

    fake.get_job = turn_the_lane
    with pytest.raises(ArchiveError) as refusal:
        run('apply', intent=planned)
    assert refusal.value.details['unverified'] == ['one']


def test_a_job_moved_to_a_third_origin_after_the_move_is_not_a_clean_receipt(cron):
    """Same defect, the simpler half: a job moved somewhere else entirely after we
    set it must be named, not counted."""
    fake = cron(job('one'))
    planned = plan()
    real_update = fake.update_job

    def then_move_it_on(job_id, updates):
        real_update(job_id, updates)
        fake.jobs[job_id]['origin'] = {'platform': 'whatsapp', 'chat_id': 'chat-third',
                                       'thread_id': None}

    fake.update_job = then_move_it_on
    with pytest.raises(ArchiveError) as refusal:
        run('apply', intent=planned)
    assert refusal.value.details['unverified'] == ['one']


def test_the_rollback_retry_reads_again_before_it_writes_again(cron):
    """Third-round P1: the retry wrote a second time from the FIRST read's verdict.
    A job that moved on between the two attempts was overwritten with our origin."""
    fake = cron(job('one'))
    planned = plan()
    fake.persist_then_raise.add('one')
    with pytest.raises(ArchiveError):
        run('apply', intent=planned)
    fake.persist_then_raise.discard('one')
    moved = {'id': 'one', 'original_origin': dict(SOURCE),
             'intended_origin': dict(MAIN), 'attempted': True, 'moved': True}
    third = {'platform': 'whatsapp', 'chat_id': 'chat-third', 'thread_id': None}
    fake.updates.clear()
    real_update = fake.update_job

    def move_it_elsewhere(job_id, updates):
        real_update(job_id, updates)
        fake.jobs[job_id]['origin'] = dict(third)

    fake.update_job = move_it_elsewhere
    answer = run('rollback', intent={'jobs': [moved], 'declined': []})
    # Exactly one write: the retry re-read first, saw a third origin, and stopped.
    assert len(fake.updates) == 1
    assert fake.jobs['one']['origin'] == third
    assert answer['jobs_moved_elsewhere'] == ['one']


def test_a_declined_archive_is_not_refused_over_a_job_it_would_not_move(cron, monkeypatch):
    """Third-round P1: the delivery-loss check ran before the declined branch, so a
    valid declined request was refused over a hypothetical move it never makes."""
    # Same fragile job as the loss test: the origin decides its delivery, and with
    # Main as the origin core resolves nothing. A reparenting request must refuse
    # it. A declined one moves nothing, so the hypothetical is never asked.
    cron(job('fragile'))
    import cron.scheduler_delivery as delivery
    real = delivery._resolve_delivery_targets

    def resolve(job_dict, *, for_failure=False):
        if (job_dict.get('origin') or {}).get('chat_id') == MAIN['chat_id']:
            return []
        return real(job_dict, for_failure=for_failure)

    monkeypatch.setattr(delivery, '_resolve_delivery_targets', resolve)
    with pytest.raises(ArchiveError) as refusal:
        plan()
    assert refusal.value.details['reason'] == 'delivery_would_be_lost'
    declined = plan(request(reparent=False))
    assert declined == {'jobs': [], 'declined': ['fragile']}
    assert run('apply', body=request(reparent=False), intent=declined) == {
        'reparented': {'cron_jobs': 0}, 'tasks_reparented': False}


def test_a_refusal_is_persisted_with_the_diagnosis_already_in_it(tmp_path, cron):
    """Third-round P2: the refusal row was written before the engine's diagnosis was
    merged, so a receipt read of an interrupted operation replayed a refusal
    stripped of the reason and the named job."""
    fake = cron(job('one'))
    fake.refuse_update.add('one')
    store = ArchiveStore(tmp_path / 'archive.db')
    with pytest.raises(ArchiveError):
        store.archive('side', request(), archive_reparent)
    with closing(sqlite3.connect(tmp_path / 'archive.db')) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT refusal_json FROM archive_operations').fetchone()
    persisted = json.loads(row['refusal_json'])
    assert persisted['reason'] == 'cron_update_refused'
    assert persisted['refused'] == ['one']


def test_core_is_asked_once_per_job_state_not_once_per_question(cron):
    """The resolver is memoised on the job content resolution depends on, so a
    changed row is a different key and never a stale answer."""
    fake = cron(job('one'), job('two'))
    calls = []
    from cron import scheduler_delivery
    real = scheduler_delivery._resolve_delivery_targets

    def counted(row, *, for_failure=False):
        calls.append((row.get('id'), row.get('origin'), for_failure))
        return real(row, for_failure=for_failure)

    def keys():
        return [json.dumps(call, sort_keys=True, default=str) for call in calls]

    scheduler_delivery._resolve_delivery_targets = counted
    try:
        planned = plan()
        # Plan asks about each job as it stands and with Main as its origin, on both
        # lanes: eight questions for two jobs, and not one of them twice. Without
        # the cache the loss check repeated the Main variant, so it was twelve.
        planned_calls = keys()
        calls.clear()
        run('apply', intent=planned)
        applied_calls = keys()
    finally:
        scheduler_delivery._resolve_delivery_targets = real
    assert [row['id'] for row in planned['jobs']] == ['one', 'two']
    assert len(planned_calls) == len(set(planned_calls)) == 8
    # The cache lives for ONE phase, so apply asks again from scratch - a job that
    # changed between the phases must not be answered from the plan's reading.
    assert len(applied_calls) == len(set(applied_calls))
