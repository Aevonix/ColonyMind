"""Vetted lesson verdicts bind only an exact adjacent answer to its direct input."""
from contextlib import closing
from datetime import timedelta

import pytest

from colonymind.mind.lessons import WATERMARK
from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash
from colonymind.work.night import WorkNight
from test_mind_consolidate import Fixture, OWNER
from test_mind_lessons_night import LessonRouter, judged, NOON
from test_work_projection import envelope, operation
from test_work_night import record

VERDICT = 'The report change is completely wrong.'


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    fixture = Fixture(tmp_path, at=NOON, config={'faculties': {
        'consolidation': False, 'lessons': True, 'work_records': True}},
        router=LessonRouter(lambda prompt: {'ops': [], 'verdicts': [judged(prompt, VERDICT, VERDICT)]}))
    yield fixture
    fixture.store.close()


def turn(fx, ident, messages, *, work=None):
    fx.ledger.record_source(ident, contact_id=OWNER, session_id='session', messages=messages,
        occurred_at=fx.now.isoformat(), derive_claims=False,
        **({'work': {'envelope': work, 'principal_id': 'host'}} if work else {}))
    fx.shift(minutes=1)


def setup(fx, *, separate=False, supplied=True, gap=None, repeated=False):
    request = {'role': 'user', 'content': 'Please update the report.'}
    reply = {'role': 'assistant', 'content': 'I updated the report.'}
    messages = [request] if separate else [request, reply]
    if repeated:
        messages.append(reply)
    turn(fx, 'root', messages, work=envelope(operations=[operation(evidence={'kind': 'none'})]))
    if separate:
        refs = [{'source_id': 'root', 'source_version': canonical_turn_digest(messages)}]
        reply['_supplied_sources'] = refs
        if supplied:
            reply['_supplied_inputs'] = [{'source_id': 'root', 'input_message_hash': source_message_hash('session', request)}]
        turn(fx, 'reply', [reply])
    if gap:
        turn(fx, 'gap', [{'role': gap, 'content': 'An intervening message.'}])
    turn(fx, 'verdict', [{'role': 'user', 'content': VERDICT}])
    worker = WorkNight(fx.ledger, OWNER)
    worker.store.reconcile()
    return worker


@pytest.mark.parametrize('options', [dict(gap='user'), dict(gap='tool'), dict(gap='system'),
                                    dict(separate=True, supplied=False), dict(repeated=True)])
async def test_ambiguous_or_non_adjacent_verdicts_remain_unlinked(fx, options):
    worker = setup(fx, **options)
    await fx.mind.consolidate()
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None


async def test_durable_handoff_precedes_progress_and_work_stage(fx, monkeypatch):
    worker = setup(fx)
    original = fx.mind.lessons._mark
    def crash(packet, now):
        with closing(worker.ledger._connect()) as conn:
            assert conn.execute("SELECT count(*) FROM work_verdict_pending WHERE state='pending'").fetchone()[0] == 1
        raise RuntimeError('interrupted before lesson cursor')
    monkeypatch.setattr(fx.mind.lessons, '_mark', crash)
    async def call(*args, **kw):
        return {'ops': [], 'verdicts': [judged(kw['user'], VERDICT, VERDICT)]}
    with pytest.raises(RuntimeError):
        await fx.mind.lessons.night(record(), fx.now, call=call,
                                  on_verdicts=worker.capture_verdicts)
    assert fx.mind.mind_state.get(WATERMARK) is None
    monkeypatch.setattr(fx.mind.lessons, '_mark', original)
    async def no_titles(*args, **kwargs):
        return None
    await WorkNight(fx.ledger, OWNER).run(record(), fx.now, call=no_titles)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'


async def test_handoff_failure_does_not_advance_lessons(fx):
    setup(fx)
    async def call(*args, **kw):
        packet = fx.mind.lessons._packet(fx.now)
        label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
        return {'ops': [], 'verdicts': [{'turn': label, 'work_was': 'wrong', 'quote': VERDICT}]}
    def fail(*args):
        raise RuntimeError('disk unavailable')
    with pytest.raises(RuntimeError):
        await fx.mind.lessons.night(record(), fx.now, call=call, on_verdicts=fail)
    assert fx.mind.mind_state.get(WATERMARK) is None


@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('projection', ['current', 'missing', 'stale'])
def test_pending_verdict_survives_harmless_refresh(fx, restart, projection):
    from colonymind.work import refresh_work, _store
    from colonymind.work.ledger import mark_dirty
    from colonymind.work.verdicts import apply_pending
    worker = setup(fx)
    before = worker.store.get_root('root', contact_id=OWNER)
    if projection == 'missing':
        with closing(worker.store._connect()) as conn, conn:
            worker.store._delete_root(conn, 'root')
    elif projection == 'stale':
        with closing(worker.ledger._connect()) as conn, conn:
            mark_dirty(conn, 'root')
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    refresh_work(fx.ledger, ['root'])
    assert _store(str(fx.ledger.db_path.parent)).wait_pending(timeout=10)
    assert worker.store.get_root('root', contact_id=OWNER)['revision'] == before['revision']
    if restart:
        worker = WorkNight(fx.ledger, OWNER)
    apply_pending(worker, record(), fx.now + timedelta(hours=1))
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'
    with closing(worker.ledger._connect()) as conn:
        assert conn.execute('SELECT state FROM work_verdict_pending').fetchone()[0] == 'linked'


def test_capture_revision_race_keeps_lesson_cursor_retryable(fx, monkeypatch):
    from colonymind.work.ledger import mark_dirty
    worker = setup(fx)
    with closing(worker.ledger._connect()) as conn, conn:
        mark_dirty(conn, 'root')
    original = worker.store.rebuild_root
    def changed_again(ident):
        original(ident)
        with closing(worker.ledger._connect()) as conn, conn:
            mark_dirty(conn, ident)
    monkeypatch.setattr(worker.store, 'rebuild_root', changed_again)
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    with pytest.raises(RuntimeError, match='work projection changed during verdict capture'):
        worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    assert fx.mind.mind_state.get(WATERMARK) is None
    with closing(worker.ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM work_verdict_pending').fetchone()[0] == 0


def test_pending_verdict_waits_for_refreshed_projection(fx, monkeypatch):
    from colonymind.work import refresh_work, _store
    from colonymind.work.verdicts import apply_pending
    worker = setup(fx)
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    monkeypatch.setattr(_store(str(fx.ledger.db_path.parent)), 'schedule', lambda: None)
    refresh_work(fx.ledger, ['root'])
    apply_pending(worker, record(), fx.now)
    with closing(worker.ledger._connect()) as conn:
        assert conn.execute('SELECT state FROM work_verdict_pending').fetchone()[0] == 'pending'
    worker.store.rebuild_root('root')
    apply_pending(worker, record(), fx.now + timedelta(hours=1))
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'


def test_pending_verdict_rejects_changed_semantic_revision(fx):
    from colonymind.work.verdicts import apply_pending
    from test_work_projection import event
    worker = setup(fx)
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    before = worker.store.get_root('root', contact_id=OWNER)
    worker.store.admit_event(event(envelope(operations=[operation('later-write')])),
                            contact_id=OWNER, principal_id='host')
    assert worker.store.wait_pending(timeout=10)
    assert worker.store.get_root('root', contact_id=OWNER)['revision'] != before['revision']
    apply_pending(worker, record(), fx.now + timedelta(hours=1))
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None
    with closing(worker.ledger._connect()) as conn:
        assert conn.execute('SELECT state FROM work_verdict_pending').fetchone()[0] == 'stale'


async def test_conflicting_model_verdict_is_not_linked(fx):
    worker = setup(fx)
    fx.router.answers['mind_lessons'] = lambda messages, context: {'ops': [], 'verdicts': [
        judged(messages[-1]['content'], VERDICT, VERDICT, 'right'),
        judged(messages[-1]['content'], VERDICT, VERDICT, 'wrong')]}
    await fx.mind.consolidate()
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None


@pytest.mark.parametrize('when', ['capture', 'apply', 'read'])
async def test_delayed_intervening_turn_revokes_link(fx, when):
    worker = setup(fx)
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    async def no_titles(*args, **kw):
        return None
    if when != 'capture':
        worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    if when == 'read':
        await worker.run(record(), fx.now, call=no_titles)
        assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'
    fx.ledger.record_source('delayed', contact_id=OWNER, session_id='session',
        messages=[{'role': 'user', 'content': 'An intervening request.'}],
        occurred_at=(NOON + timedelta(seconds=30)).isoformat(), derive_claims=False)
    if when == 'capture':
        worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    await worker.run(record(), fx.now + timedelta(hours=1), call=no_titles)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None


async def test_tied_reply_sources_are_ambiguous(fx):
    worker = setup(fx)
    fx.ledger.record_source('tied', contact_id=OWNER, session_id='session',
        messages=[{'role': 'user', 'content': 'Another request.'}, {'role': 'assistant', 'content': 'Another reply.'}],
        occurred_at=NOON.isoformat(), derive_claims=False)
    await fx.mind.consolidate()
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None


async def test_coverage_limits_are_reported(fx):
    worker = setup(fx)
    for i in range(7):
        for j in range(10):
            fx.turn(f'other-{i}-{j}', OWNER, f'session-{i}', f'Request {j}', 'Reply')
            fx.shift(minutes=1)
    # This bounded packet intentionally omits the older root session.
    fx.router.answers['mind_lessons'] = {'ops': [], 'verdicts': []}
    result = await fx.mind.consolidate()
    summary = worker.store.summary(contact_id=OWNER)
    coverage = summary['verdict_coverage']
    assert coverage['session_limit'] == 6 and coverage['turn_limit'] == 8
    assert coverage['sessions_reviewed'] == 6 and coverage['sessions_omitted'] == 2
    assert coverage['sessions_with_omitted_turns'] == 6
    assert result['counts']['work_verdict_sessions_omitted'] == 2


async def test_pending_projection_retries_and_old_verdict_cannot_replace_newer(fx):
    from colonymind.work.verdicts import apply_pending
    worker = setup(fx)
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    with closing(worker.store._connect()) as conn, conn:
        worker.store._delete_root(conn, 'root')
    apply_pending(worker, record(), fx.now)
    with closing(worker.ledger._connect()) as conn:
        assert conn.execute('SELECT state,attempts FROM work_verdict_pending').fetchone()[:] == ('pending', 1)
    worker.store.rebuild_root('root')
    apply_pending(worker, record(), fx.now)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None
    later = fx.now + timedelta(hours=1)
    apply_pending(worker, record(), later)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'
    row = worker.store.get_root('root', contact_id=OWNER)
    assert worker.store.write_derived('root', contact_id=OWNER, expected_revision=row['revision'],
        overlay='confirmed', verdict_order=(later.isoformat(), 'later-verdict'))
    with closing(worker.ledger._connect()) as conn, conn:
        conn.execute("UPDATE work_verdict_pending SET state='pending',next_attempt=0")
    apply_pending(worker, record(), later)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'confirmed'


def test_crash_after_overlay_write_preserves_canonical_decision(fx, monkeypatch):
    from colonymind.work.ledger import mark_dirty
    from colonymind.work.verdicts import apply_pending
    worker = setup(fx)
    packet = fx.mind.lessons._packet(fx.now)
    label = next(key for key, item in packet['owner'].items() if item['text'] == VERDICT)
    worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    original = worker.store.write_derived
    def interrupted(*args, **kwargs):
        assert original(*args, **kwargs)
        raise RuntimeError('crash before projection acknowledgment')
    monkeypatch.setattr(worker.store, 'write_derived', interrupted)
    with pytest.raises(RuntimeError):
        apply_pending(worker, record(), fx.now)
    with closing(fx.ledger._connect()) as conn, conn:
        assert conn.execute('SELECT state FROM work_verdict_pending').fetchone()[0] == 'accepted'
        mark_dirty(conn, 'root')
    worker.store.rebuild_root('root')
    worker.store.path.unlink()
    repaired = WorkNight(fx.ledger, OWNER)
    repaired.store.reconcile()
    assert repaired.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'
    apply_pending(repaired, record(), fx.now + timedelta(hours=1))
    with closing(fx.ledger._connect()) as conn:
        assert conn.execute('SELECT state FROM work_verdict_pending').fetchone()[0] == 'linked'

from colonymind.memory import curated

@pytest.mark.parametrize('separate', [False, True])
async def test_exact_judged_reply_links_and_survives_restart(fx, separate):
    worker = setup(fx, separate=separate)
    result = await fx.mind.consolidate()
    assert result['counts']['work_verdicts_linked'] == 1
    root = worker.store.get_root('root', contact_id=OWNER)
    assert root['overlay'] == 'corrected' and root['status'] == 'unproven'
    # Titles can fail independently; the verified overlay is retained.
    assert root['title'] is None
    reopened = WorkNight(fx.ledger, OWNER)
    assert reopened.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(fx.ledger)
    _source_forget_receipt = curated.forget_sources(fx.ledger, OWNER, source_ids=['verdict'])
    assert historical_bytes(fx.ledger) == _native_before_forget
    assert all(curated.is_forgotten(fx.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert reopened.store.get_root('root', contact_id=OWNER)['overlay'] is None


async def test_queued_verdict_fenced_by_erasure_and_root_revision(fx):
    worker = setup(fx)
    packet = fx.mind.lessons._packet(fx.now)
    label = next((key for key, item in packet['owner'].items() if item['text'] == VERDICT))
    worker.capture_verdicts(packet, {label: 'wrong'}, record(), fx.now)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=['verdict'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal

    async def no_titles(*args, **kwargs):
        return None
    await WorkNight(fx.ledger, OWNER).run(record(), fx.now, call=no_titles)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] is None
    with closing(worker.ledger._connect()) as conn:
        assert conn.execute('SELECT state FROM work_verdict_pending').fetchone()[0] == 'stale'

@pytest.mark.parametrize('erase', [False, True])
@pytest.mark.parametrize('refresh', [False, True])
async def test_projection_loss_rehydrates_only_current_vetted_verdict(fx, erase, refresh):
    worker = setup(fx)
    await fx.mind.consolidate()
    watermark = fx.mind.mind_state.get(WATERMARK)
    assert worker.store.get_root('root', contact_id=OWNER)['overlay'] == 'corrected'
    if erase:
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(fx.ledger)
        curated.forget_sources(fx.ledger, OWNER, source_ids=['verdict'])
        assert historical_bytes(fx.ledger) == _native_before_withdrawal
        from colonymind.work import _store
        _store(str(fx.ledger.db_path.parent)).wait_pending()
    if refresh:
        from colonymind.work.ledger import mark_dirty
        with closing(fx.ledger._connect()) as conn, conn:
            mark_dirty(conn, 'root')
        worker.store.rebuild_root('root')
    worker.store.path.unlink()
    repaired = WorkNight(fx.ledger, OWNER)
    repaired.store.reconcile()
    assert repaired.store.get_root('root', contact_id=OWNER)['overlay'] == (None if erase else 'corrected')
    assert fx.mind.mind_state.get(WATERMARK) == watermark
