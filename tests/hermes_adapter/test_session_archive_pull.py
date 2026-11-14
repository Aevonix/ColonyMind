"""Real core reparenting, journal replay and a sidecar with no Hermes install."""
import asyncio
from contextlib import closing, contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from conftest import OWNER, ROOT
from colonymind_hermes.archive import ArchiveWorker, verify_owner
from colonymind_hermes.archive_core.store import ArchiveError
from colonymind_hermes.capture import SessionMap
from colonymind_hermes.client import ColonyMindClient, Settings, load_settings

SOURCE = {'platform': 'telegram', 'chat_id': 'archive-side', 'thread_id': None}
MAIN = {'platform': 'telegram', 'chat_id': 'archive-main', 'thread_id': None}

# Each probe below runs under this interpreter, whichever Python the suite runs on. Where
# COLONYMIND_TEST_HERMES_PYTHON names a second interpreter with Hermes installed (a Python 3.14 environment), the
# same probes run under that one as well; elsewhere that parameter is skipped as a missing live qualification
# prerequisite (conftest.LIVE_PREREQUISITES).
INTERPRETERS = ['this-interpreter', pytest.param('second-interpreter', marks=pytest.mark.requires_live_qualification)]


def probe_python(interpreter):
    return sys.executable if interpreter == 'this-interpreter' else os.environ['COLONYMIND_TEST_HERMES_PYTHON']


def probe_version(interpreter):
    """The Python version a probe must find itself running on."""
    return '%d.%d' % sys.version_info[:2] if interpreter == 'this-interpreter' else '3.14'


@pytest.mark.parametrize('interpreter', INTERPRETERS)
@pytest.mark.parametrize('mode,options', [('scope-run', []), ('scope-run', ['missing']),
    ('scope-run', ['failure']), ('scope-run', ['missing', 'failure']),
    ('unload-registration', []), ('discovery', [])])
def test_reviewed_profile_unload_and_discovery_reproductions(tmp_path, interpreter, mode, options):
    from cron import jobs
    result = subprocess.run([probe_python(interpreter), '-I', '-B',
                             str(Path(__file__).with_name('archive_behavior_probe.py')),
                             str(Path(jobs.__file__).resolve().parents[1]), mode, str(tmp_path / 'probe'),
                             *options], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('interpreter', INTERPRETERS)
@pytest.mark.parametrize('lane', ['success', 'failure'])
@pytest.mark.parametrize('stage', ['prep', 'commit', 'replay'])
def test_routing_drift_compensates_without_positive_receipt(tmp_path, interpreter, lane, stage):
    from cron import jobs
    result = subprocess.run([probe_python(interpreter), '-I', '-B',
                             str(Path(__file__).with_name('routing_drift_probe.py')),
                             str(Path(jobs.__file__).resolve().parents[1]), stage + '-' + lane, str(tmp_path / 'probe')],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_unload_while_gateway_resolver_returns_never_launches_child(home, monkeypatch):
    from cron import jobs
    from colonymind_hermes import archive
    settings, plan = make_plan(home, monkeypatch)
    jobs.create_job('delayed resolver', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    entered, release = threading.Event(), threading.Event()
    def resolve(job):
        entered.set()
        assert release.wait(3)
        return []
    monkeypatch.setattr(archive, '_resolver', lambda: resolve)
    def never_launch(*args, **kwargs):
        pytest.fail('delayed preparation launched a child after close')
    monkeypatch.setattr(archive.subprocess, 'Popen', never_launch)
    work = worker._engine_unit(plan)
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        with archive._checked_jobs_lock():
            pass  # Stalled profile/plugin resolution must not block ordinary cron.
        worker.close()
    finally:
        release.set()
    with pytest.raises(ArchiveError, match='busy'):
        await work
    assert not worker.engine_running


@pytest.mark.parametrize('change', ['new_job', 'changed_lane'])
def test_cron_change_during_gateway_resolution_cannot_certify_stale_targets(home, monkeypatch, change):
    from cron import jobs
    from colonymind_hermes import archive
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('original delivery', 'every 1h', deliver='origin', origin=SOURCE)
    original = archive._checked_jobs_lock
    @contextmanager
    def competing_writer():
        if change == 'new_job':
            jobs.create_job('new delivery', 'every 1h', deliver='origin', origin=SOURCE)
        else:
            jobs.update_job(job['id'], {'failure_deliver': 'origin'})
        with original() as fd:
            yield fd
    monkeypatch.setattr(archive, '_checked_jobs_lock', competing_writer)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    with pytest.raises(ArchiveError, match='busy'):
        worker._apply_bounded(plan)
    assert all(row['origin'] == SOURCE for row in jobs.list_jobs())


@pytest.mark.parametrize('interpreter', INTERPRETERS)
@pytest.mark.parametrize('mode', ['park', 'park-rollback'])
def test_actual_gateway_sigkill_stops_unit_and_recovers(tmp_path, interpreter, mode):
    from cron import jobs
    scratch = tmp_path / 'probe'
    log = (tmp_path / 'gateway.log').open('w')
    process = subprocess.Popen([probe_python(interpreter), '-I', '-B',
                                str(Path(__file__).with_name('archive_behavior_probe.py')),
                                str(Path(jobs.__file__).resolve().parents[1]), mode, str(scratch)],
                               stdout=log, stderr=log)
    child = None
    try:
        marker = scratch / 'parked.json'
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            assert process.poll() is None
            try:
                child = json.loads(marker.read_text())['pid']
                break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(.01)
        assert child is not None, 'unit did not publish its parked PID'
        process.kill()
        process.wait(timeout=2)
        # Parent-death termination must beat the five-second unit deadline.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(.01)
        else:
            pytest.fail('unit survived gateway SIGKILL or was not reaped')
        for path in (scratch / 'hermes/cron/.jobs.lock',
                     scratch / 'hermes/state/outbox-archive.sqlite3.lock',
                     scratch / 'hermes/state/outbox-archive.sqlite3.worker.lock'):
            with path.open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reuse the probe's durable authority and real moved records for recovery.
        from colonymind_hermes import archive
        hermes, instance = scratch / 'hermes', scratch / 'instance'
        settings = Settings('http://unused', instance / 'api.key', 'synthetic', instance,
                            hermes, hermes / 'state/outbox.sqlite3')
        client = ColonyMindClient(settings)
        worker = archive.ArchiveWorker(settings, client, SessionMap(settings, client))
        plan = json.loads((scratch / 'plan.json').read_text())
        with jobs.use_cron_store(hermes):
            receipt = worker._apply_bounded(plan)
            assert receipt['error'] == 'uncertain' and not receipt['half_applied']
            assert receipt['restored'] == {'cron_jobs': 2 if mode == 'park-rollback' else 1}
            path = jobs._current_cron_store().jobs_file
            before = path.read_bytes(), path.stat().st_mtime_ns
            assert worker._apply_bounded({**plan, 'claim': 'reclaimed'}) == receipt
            assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()


def make_plan(home, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(home.hermes))
    monkeypatch.setenv('COLONYMIND_HOME', str(home.instance))
    monkeypatch.setenv('HERMES_DISABLE_LAZY_INSTALLS', '1')
    settings = load_settings()
    with closing(sqlite3.connect(home.hermes / 'state.db')) as db:
        db.execute('CREATE TABLE gateway_routing (entry_json TEXT)')
        for ident, origin in (('side', SOURCE), ('main', MAIN)):
            entry = {'session_id': ident, 'origin': {**origin, 'user_id': '1001', 'chat_type': 'dm'}}
            db.execute('INSERT INTO gateway_routing VALUES (?)', (json.dumps(entry),))
        db.commit()
    request = {'operation_id': 'op', 'archived': True, 'reparent_to_main': True,
               'main_session_id': 'main', 'source_origin': SOURCE, 'main_origin': MAIN}
    return settings, {'operation_id': 'op', 'destination_id': 'side', 'request': request,
                      'authority': {'speaks_for_owner': True, 'sessions:write': True, 'owner_id': OWNER},
                      'created_at': '2026-10-19T12:00:00+00:00', 'claim': 'first-claim'}


def test_real_jobs_and_durable_receipt_replay_after_reclaim(home, monkeypatch):
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('archive test', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    result = worker.apply(plan)
    assert result['reparented'] == {'cron_jobs': 1}
    assert jobs.get_job(job['id'])['origin'] == MAIN
    def never_again(*args, **kwargs):
        raise AssertionError('reclaimed operation attempted a second cron update')
    monkeypatch.setattr(jobs, 'update_job', never_again)
    restarted = ArchiveWorker(settings, client, SessionMap(settings, client))
    assert restarted.apply({**plan, 'claim': 'second-claim'}) == result
    assert worker.path.stat().st_mode & 0o777 == 0o600


def test_real_core_refusal_compensates_and_replays_by_name(home, monkeypatch):
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    records = [jobs.create_job('rollback test', 'every 1h', deliver='origin', origin=SOURCE) for _ in range(2)]
    refused_id = sorted(row['id'] for row in records)[1]
    original = jobs.update_job
    def refuse(ident, updates):
        if ident == refused_id and updates.get('origin') == MAIN:
            raise ValueError('core refused update')
        return original(ident, updates)
    monkeypatch.setattr(jobs, 'update_job', refuse)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    result = worker.apply(plan)
    assert result['error'] == 'uncertain'
    assert result['refused'] == [refused_id] and result['reason'] == 'cron_update_refused'
    # Both attempted jobs are proved back at source, including the refused one.
    assert result['restored'] == {'cron_jobs': 2} and not result['half_applied']
    assert all(jobs.get_job(row['id'])['origin'] == SOURCE for row in records)
    assert worker.apply({**plan, 'claim': 'reclaimed'}) == result


def test_crash_after_real_cron_write_recovers_by_compensation(home, monkeypatch):
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('crash test', 'every 1h', deliver='origin', origin=SOURCE)
    original = jobs.update_job
    def crash(ident, changes):
        result = original(ident, changes)
        if changes.get('origin') == MAIN:
            raise SystemExit('simulated gateway interruption')
        return result
    monkeypatch.setattr(jobs, 'update_job', crash)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    with pytest.raises(SystemExit):
        worker.apply(plan)
    assert jobs.get_job(job['id'])['origin'] == MAIN
    result = worker.apply({**plan, 'claim': 'reclaimed'})
    assert result['error'] == 'uncertain' and result['restored'] == {'cron_jobs': 1}
    assert jobs.get_job(job['id'])['origin'] == SOURCE


def test_journal_receipt_survives_transport_cache_failure_and_ownership_drift(home, monkeypatch):
    from cron import jobs
    from colonymind_hermes import archive
    from colonymind_hermes.archive_core.store import ArchiveStore
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('receipt gap test', 'every 1h', deliver='origin', origin=SOURCE)
    original = ArchiveStore._writer
    calls = 0
    @contextmanager
    def fail_transport_record(self):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError('transport record unavailable after engine commit')
        with original(self) as db:
            yield db
    monkeypatch.setattr(ArchiveStore, '_writer', fail_transport_record)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    with pytest.raises(OSError):
        worker.apply(plan)
    assert jobs.get_job(job['id'])['origin'] == MAIN
    monkeypatch.setattr(ArchiveStore, '_writer', original)
    with closing(sqlite3.connect(worker.path)) as db:
        expected = json.loads(db.execute('SELECT receipt_json FROM archive_operations').fetchone()[0])
    def no_new_operation(*args, **kwargs):
        raise AssertionError('an applied operation was revalidated or mutated')
    monkeypatch.setattr(archive, 'verify_owner', no_new_operation)
    monkeypatch.setattr(jobs, 'update_job', no_new_operation)
    assert worker.apply({**plan, 'claim': 'reclaimed'}) == expected


@pytest.mark.parametrize('mutation', ['grant', 'scope', 'owner', 'origin', 'session', 'sender'])
def test_gateway_owner_check_refuses_by_name_without_writes(home, monkeypatch, mutation):
    settings, plan = make_plan(home, monkeypatch)
    if mutation in {'grant', 'scope', 'owner'}:
        field = {'grant': 'speaks_for_owner', 'scope': 'sessions:write', 'owner': 'owner_id'}[mutation]
        plan['authority'][field] = False
    elif mutation == 'origin':
        plan['request']['source_origin'] = {**SOURCE, 'chat_id': 'unowned-chat'}
    elif mutation == 'session':
        plan['destination_id'] = 'unowned-session'
    else:
        with closing(sqlite3.connect(home.hermes / 'state.db')) as db:
            db.execute('UPDATE gateway_routing SET entry_json=replace(entry_json,?,?)', ('1001', '2002'))
            db.commit()
    with pytest.raises(ArchiveError) as refused:
        verify_owner(plan, settings, SessionMap(settings, ColonyMindClient(settings)))
    assert refused.value.code == 'unauthorized'
    assert refused.value.details['refused'] == [plan['destination_id']]
    assert not (home.hermes / 'cron' / 'jobs.json').exists()


@pytest.mark.asyncio
async def test_supervised_worker_stops_on_unload_and_old_sidecar_backs_off(home, sidecar, monkeypatch):
    settings, _ = make_plan(home, monkeypatch)
    worker = ArchiveWorker(settings, ColonyMindClient(settings), SessionMap(settings, ColonyMindClient(settings)))
    callbacks, tasks = [], []
    def spawn(coro, *, name):
        task = asyncio.create_task(coro, name=name)
        tasks.append(task)
        return task
    def on_unload(callback):
        callbacks.append(callback)
        return SimpleNamespace(active=True)
    worker.start(SimpleNamespace(on_unload=on_unload, spawn_task=spawn))
    for _ in range(100):
        if sidecar.calls('/v1/host/sessions/archive/plans'):
            break
        await asyncio.sleep(.01)
    assert len(sidecar.calls('/v1/host/sessions/archive/plans')) == 1
    await asyncio.sleep(.1)
    assert len(sidecar.calls('/v1/host/sessions/archive/plans')) == 1
    assert len(tasks) == 1 and len(callbacks) == 1
    callbacks[0]()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert tasks[0].done() and worker.closed and worker.lease is None


@pytest.mark.asyncio
async def test_repeated_unload_cancellation_keeps_the_lease_until_apply_finishes(home, sidecar, monkeypatch):
    settings, plan = make_plan(home, monkeypatch)
    original = sidecar.dispatch
    def dispatch(method, path, query, payload):
        if path == '/v1/host/sessions/archive/plans':
            return 200, {'plan': plan}
        return original(method, path, query, payload)
    monkeypatch.setattr(sidecar, 'dispatch', dispatch)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    started, release = threading.Event(), threading.Event()
    def apply(_plan):
        started.set()
        assert release.wait(5)
        return {}
    monkeypatch.setattr(worker, '_apply_bounded', apply)
    task = asyncio.create_task(worker.run())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(.02)
        task.cancel()
        await asyncio.sleep(.02)
        assert task.done() and worker.lease is not None
        with open(str(worker.path) + '.worker.lock', 'rb') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for _ in range(100):
            if worker.lease is None:
                break
            await asyncio.sleep(.01)
    assert worker.lease is None


@pytest.mark.parametrize('phase', ['forward', 'rollback'])
def test_ordinary_cron_writer_waits_for_the_whole_archive_unit(home, monkeypatch, phase):
    from cron import jobs
    from colonymind_hermes import archive
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('competing writer', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    other = {**MAIN, 'chat_id': 'independent-edit'}
    started, finished = threading.Event(), threading.Event()
    original_update, original_engine = jobs.update_job, archive.archive_reparent
    writer = None
    engine_thread = threading.current_thread()
    def compete():
        started.set()
        original_update(job['id'], {'origin': other})
        finished.set()
    def update(ident, changes):
        nonlocal writer
        target = MAIN if phase == 'forward' else SOURCE
        if threading.current_thread() is engine_thread and changes.get('origin') == target:
            writer = threading.Thread(target=compete)
            writer.start()
            assert started.wait(2)
            assert not finished.wait(.05), 'ordinary cron writer escaped the engine critical section'
        return original_update(ident, changes)
    def engine(**kwargs):
        if phase == 'rollback' and kwargs['phase'] == 'commit':
            raise RuntimeError('force compensation')
        return original_engine(**kwargs)
    monkeypatch.setattr(jobs, 'update_job', update)
    monkeypatch.setattr(archive, 'archive_reparent', engine)
    result = worker.apply(plan)
    assert writer is not None
    writer.join(3)
    assert finished.is_set()
    assert jobs.get_job(job['id'])['origin'] == other
    if phase == 'rollback':
        assert result['error'] == 'uncertain' and not result['half_applied']
        assert result['restored'] == {'cron_jobs': 1}
    else:
        assert result['reparented'] == {'cron_jobs': 1}


def test_competing_process_cron_lock_fails_closed_even_when_stock_would_continue(home, monkeypatch):
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('cross process writer', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    code = 'import fcntl,sys; f=open(sys.argv[1],"a+"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); sys.stdin.read()'
    process = subprocess.Popen([sys.executable, '-c', code, str(jobs._jobs_lock_file())],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'locked'
        monkeypatch.setattr(jobs, '_JOBS_LOCK_TIMEOUT_SECONDS', 0)
        with pytest.raises(ArchiveError, match='busy'):
            worker.apply(plan)
        assert not worker.path.exists()
        assert jobs.get_job(job['id'])['origin'] == SOURCE
    finally:
        process.communicate('', timeout=3)


@pytest.mark.parametrize('drift', ['missing_binding', 'sender_failure', 'authority_revoked'])
def test_interrupted_intent_recovers_before_current_authorization(home, monkeypatch, drift):
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('interrupted owner check', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    sessions = SessionMap(settings, client)
    worker = ArchiveWorker(settings, client, sessions)
    original = jobs.update_job
    def crash(ident, changes):
        result = original(ident, changes)
        if changes.get('origin') == MAIN:
            raise SystemExit('crash after write')
        return result
    monkeypatch.setattr(jobs, 'update_job', crash)
    with pytest.raises(SystemExit):
        worker.apply(plan)
    if drift == 'missing_binding':
        with closing(sqlite3.connect(home.hermes / 'state.db')) as db:
            db.execute('DELETE FROM gateway_routing')
            db.commit()
    elif drift == 'sender_failure':
        def unavailable(*args):
            raise RuntimeError('transient sender lookup failure')
        monkeypatch.setattr(sessions, 'sender_is_owner', unavailable)
    else:
        original_owner = type(settings).owner_contact_id
        monkeypatch.setattr(type(settings), 'owner_contact_id', lambda self: 'new-owner')
    result = worker.apply({**plan, 'claim': 'reclaimed'})
    assert result['error'] == 'uncertain'
    with closing(sqlite3.connect(worker.path)) as db:
        saved = json.loads(db.execute('SELECT refusal_json FROM archive_operations').fetchone()[0])
    if drift == 'authority_revoked':
        assert result['half_applied'] and result['receipt_incomplete']
        assert result['jobs_left_moved'] == saved['jobs_left_moved'] == [job['id']]
        assert jobs.get_job(job['id'])['origin'] == MAIN
        monkeypatch.setattr(type(settings), 'owner_contact_id', original_owner)
        result = worker.apply({**plan, 'claim': 'recover-again'})
    assert not result['half_applied'] and result['restored'] == {'cron_jobs': 1}
    assert jobs.get_job(job['id'])['origin'] == SOURCE


def test_abandoned_real_plugin_context_cannot_start_a_worker(home, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    settings, _ = make_plan(home, monkeypatch)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    ctx = PluginContext(PluginManifest(name='archive-abandoned'), PluginManager())
    ctx._abandon_load()
    def forbidden(*args, **kwargs):
        raise AssertionError('abandoned load started a task or subscription')
    monkeypatch.setattr(ctx, 'spawn_task', forbidden)
    from agent.monitoring.emitter import get_emitter
    monkeypatch.setattr(get_emitter(), 'subscribe', forbidden)
    worker.start(ctx)
    assert worker.task is worker.thread is worker.emitter is worker.lease is None
    assert not worker.path.exists()


def test_crash_between_recorded_authority_and_engine_intent_can_retry(home, monkeypatch):
    from cron import jobs
    from colonymind_hermes import archive
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('authority gap', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    original = archive.archive_reparent
    def crash(**kwargs):
        if kwargs['phase'] == 'plan':
            raise SystemExit('authority saved, intent not yet written')
        return original(**kwargs)
    monkeypatch.setattr(archive, 'archive_reparent', crash)
    with pytest.raises(SystemExit):
        worker.apply(plan)
    monkeypatch.setattr(archive, 'archive_reparent', original)
    assert worker.apply(plan)['reparented'] == {'cron_jobs': 1}
    assert jobs.get_job(job['id'])['origin'] == MAIN


def test_unload_registration_disposed_during_startup_cleans_subscription_before_spawn(home, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from agent.monitoring.emitter import get_emitter
    settings, _ = make_plan(home, monkeypatch)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    ctx = PluginContext(PluginManifest(name='archive-startup'), PluginManager())
    subscribed = []
    def subscribe(callback):
        subscribed.append(callback)
        worker.registration.dispose()
    monkeypatch.setattr(get_emitter(), 'subscribe', subscribe)
    monkeypatch.setattr(get_emitter(), 'unsubscribe', subscribed.remove)
    worker.start(ctx)
    assert worker.closed and not subscribed
    assert worker.task is worker.thread is worker.emitter is worker.lease is None
    assert not worker.path.exists()


def test_core_import_probe_finishes_before_background_startup(home, sidecar, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from agent.monitoring.emitter import get_emitter
    from colonymind_hermes import archive
    settings, _ = make_plan(home, monkeypatch)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    ctx = PluginContext(PluginManifest(name='archive-import-order'), PluginManager())
    initialized = threading.Event()
    def core():
        initialized.set()
        return (lambda: None,) * 3
    monkeypatch.setattr(archive, '_core', core)
    monkeypatch.setattr(archive, '_resolver', lambda: None)
    original = get_emitter().subscribe
    def subscribe(callback):
        assert initialized.is_set(), 'background startup raced the core import probe'
        original(callback)
    monkeypatch.setattr(get_emitter(), 'subscribe', subscribe)
    worker.start(ctx)
    assert worker.thread is not None and initialized.is_set()
    worker.close()
    worker.thread.join(2)
    assert not worker.thread.is_alive() and worker.lease is None


@pytest.mark.parametrize('interpreter', INTERPRETERS)
@pytest.mark.parametrize('running_loop', [False, True])
def test_stalled_engine_does_not_join_at_interpreter_shutdown(tmp_path, interpreter, running_loop):
    code = r'''
import asyncio, importlib.util, pathlib, sys, threading
from types import SimpleNamespace
assert sys.version.startswith(sys.argv[3])
spec = importlib.util.spec_from_file_location('colonymind_hermes', pathlib.Path(sys.argv[1]) / '__init__.py', submodule_search_locations=[sys.argv[1]])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
from colonymind_hermes import archive
archive._core = lambda: (lambda: None,) * 3
archive._resolver = lambda: None
class HTTP:
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, *args, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: {'plan': {}})
archive.httpx.AsyncClient = HTTP
worker = archive.ArchiveWorker(SimpleNamespace(outbox_path=pathlib.Path(sys.argv[2])/'outbox.db'),
                               SimpleNamespace(url='http://unused', headers=lambda: {}), None)
started = threading.Event()
def stalled(plan):
    started.set()
    threading.Event().wait()
worker._apply_bounded = stalled
ctx = SimpleNamespace(on_unload=lambda cb: SimpleNamespace(active=True), spawn_task=lambda coro, **kw: asyncio.create_task(coro))
if sys.argv[4] == 'True':
    async def main():
        worker.start(ctx)
        for _ in range(300):
            if started.is_set(): break
            await asyncio.sleep(.01)
        assert started.is_set(), 'engine never started'
        worker.close()
    asyncio.run(main())
else:
    worker.start(ctx)
    assert started.wait(3), 'engine never started'
    worker.close()
print('shutdown reached', flush=True)
'''
    env = {k: v for k, v in os.environ.items() if k != 'HERMES_KANBAN_TASK'}
    result = subprocess.run([probe_python(interpreter), '-I', '-B', '-c', code, str(ROOT / 'plugins/hermes-plugin'),
                             str(tmp_path), probe_version(interpreter), str(running_loop)],
                            capture_output=True, text=True, env=env, timeout=6)
    assert result.returncode == 0, result.stderr
    assert 'shutdown reached' in result.stdout


@pytest.mark.parametrize('interpreter', INTERPRETERS)
def test_scheduler_claim_during_stalled_archive_exits_and_recovers(home, monkeypatch, interpreter):
    settings, plan = make_plan(home, monkeypatch)
    plan_file = home.root / 'plan.json'
    plan_file.write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items() if k != 'HERMES_KANBAN_TASK'}
    result = subprocess.run([probe_python(interpreter), '-I', '-B',
                             str(Path(__file__).with_name('archive_shutdown_probe.py')),
                             str(ROOT / 'plugins/hermes-plugin'), str(plan_file), probe_version(interpreter)],
                            env=env, cwd=home.root, capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'shutdown reached: closed=True engine_running=True' in result.stdout
    assert 'claim drained after reap' in result.stdout
    # Reaping precedes unlocking: the parked mutator is dead when the process
    # exits, and a next-start replay compensates the actual persisted cron move.
    child_pid = int(plan_file.with_suffix('.parked').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    recovered = worker.apply({**plan, 'claim': 'after-deadline'})
    assert recovered['error'] == 'uncertain'
    assert recovered['restored'] == {'cron_jobs': 1} and not recovered['half_applied']
    from cron import jobs
    target, = [job for job in jobs.list_jobs() if job['name'] == 'archive target']
    assert target['origin'] == SOURCE


@pytest.mark.asyncio
async def test_supervised_unit_returns_durable_receipt(home, monkeypatch):
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('finite supervised unit', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    result = await worker._engine_unit(plan)
    assert result['reparented'] == {'cron_jobs': 1}
    assert jobs.get_job(job['id'])['origin'] == MAIN
    assert worker.apply({**plan, 'claim': 'reclaimed'}) == result


@pytest.mark.asyncio
async def test_child_owns_deadline_and_lease_if_gateway_reference_is_lost(home, monkeypatch):
    from cron import jobs
    from colonymind_hermes import archive
    settings, plan = make_plan(home, monkeypatch)
    jobs.create_job('child deadline', 'every 1h', deliver='origin', origin=SOURCE)
    client = ColonyMindClient(settings)
    worker = ArchiveWorker(settings, client, SessionMap(settings, client))
    worker.path.parent.mkdir(parents=True, exist_ok=True)
    lease_path = str(worker.path) + '.worker.lock'
    worker.lease = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(worker.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = home.root / 'child-parked'
    injection = '''
from _colonymind_archive_unit import archive as unit
from pathlib import Path
import threading
original = unit.archive_reparent
def park(**kwargs):
    if kwargs['phase'] == 'commit':
        Path(%r).touch()
        threading.Event().wait()
    return original(**kwargs)
unit.archive_reparent = park
''' % str(marker)
    monkeypatch.setattr(archive, '_UNIT_BOOTSTRAP', archive._UNIT_BOOTSTRAP.replace(
        'try:\n    response', injection + '\ntry:\n    response'))
    processes = []
    original_popen = subprocess.Popen
    class LateSupervisor(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            processes.append(self)

        def communicate(self, input=None, timeout=None):
            # Disable the parent's deadline as a fault injection. The child must
            # terminate itself; the test still reaps it rather than orphaning it.
            return super().communicate(input, timeout=timeout + 3)
    monkeypatch.setattr(archive.subprocess, 'Popen', LateSupervisor)
    work = worker._engine_unit(plan)
    try:
        for _ in range(400):
            if marker.exists():
                break
            await asyncio.sleep(.01)
        assert marker.exists(), 'child did not reach its real cron write'
        worker.close()
        # Model gateway exit closing its copy. The child's inherited open-file
        # description must continue to exclude a new worker until termination.
        with worker.lifecycle:
            worker._release_lease()
        with open(lease_path, 'rb') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ArchiveError, match='busy'):
            await work
        assert processes[0].returncode == -signal.SIGALRM
        with open(lease_path, 'rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        await asyncio.gather(work, return_exceptions=True)
        worker._release_lease()


def test_separate_interpreters_sidecar_has_no_hermes_and_worker_completes_archive(home, monkeypatch, tmp_path):
    # A sidecar-only environment: where none is named this test is skipped as a missing live qualification
    # prerequisite (conftest.LIVE_PREREQUISITES).
    sidecar_python = os.environ['COLONYMIND_TEST_SIDECAR_PYTHON']
    from cron import jobs
    settings, plan = make_plan(home, monkeypatch)
    job = jobs.create_job('archive integration', 'every 1h', deliver='origin', origin=SOURCE)
    plan['request']['operation_id'] = plan['operation_id'] = 'op/with..#?'
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen()
    url = 'http://127.0.0.1:' + str(sock.getsockname()[1])
    home.write_config(plugins={'enabled': ['colonymind'], 'hook_callback_timeout': 0,
                              'colonymind': {'sidecar_url': url, 'key_file': str(home.key_file)}})
    env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'COLONYMIND_API_KEY'}}
    env.update(COLONYMIND_OWNER_PERSON_ID=OWNER, ARCHIVE_TEST_DB=str(tmp_path / 'overlay.db'),
               ARCHIVE_TEST_FD=str(sock.fileno()), COLONYMIND_BODY_THREAD='0',
               COLONYMIND_SKIP_DOTENV='1', PYTHON_DOTENV_DISABLED='1')
    # Test credential stays in the environment; no command line or output key.
    env['ARCHIVE_TEST_CREDENTIAL'] = 'archive-test-local-credential'
    server = r'''
import importlib.util, os
assert importlib.util.find_spec('cron') is None
from fastapi import FastAPI
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.api.archive_queue import ArchiveQueueClient
from colonymind.session_archive.store import ArchiveStore
import colonymind, uvicorn
print(colonymind.__file__, flush=True)
store = ArchiveStore(os.environ['ARCHIVE_TEST_DB'])
host.set_session_archive_store(store)
host.set_session_archive_engine(ArchiveQueueClient(store))
app = FastAPI()
app.add_middleware(ApiKeyMiddleware, api_key=os.environ['ARCHIVE_TEST_CREDENTIAL'])
app.include_router(host.router)
uvicorn.run(app, fd=int(os.environ['ARCHIVE_TEST_FD']), log_level='error', timeout_graceful_shutdown=1)
'''
    log = (tmp_path / 'sidecar.log').open('w+')
    process = subprocess.Popen([sidecar_python, '-I', '-B', '-c', server], env=env, cwd=tmp_path,
                               pass_fds=(sock.fileno(),), stdout=log, stderr=log)
    worker_process = None
    try:
        headers = {'Authorization': 'Bearer ' + env['ARCHIVE_TEST_CREDENTIAL']}
        with httpx.Client(base_url=url, headers=headers, timeout=20, trust_env=False) as api:
            for _ in range(200):
                assert process.poll() is None, 'sidecar-only interpreter failed to start'
                try:
                    response = api.get('/v1/host/sessions/archived', timeout=.2)
                    break
                except httpx.TransportError:
                    time.sleep(.05)
            assert response.status_code == 409
            worker_env = dict(env, COLONYMIND_API_KEY=env['ARCHIVE_TEST_CREDENTIAL'], ARCHIVE_TEST_URL=url)
            worker_code = r'''
import asyncio
from hermes_cli.plugins import get_plugin_manager
async def main():
    manager = get_plugin_manager()
    manager.discover_and_load(force=True)
    loaded = manager._plugins['colonymind']
    assert loaded.enabled, loaded.error
    await asyncio.Event().wait()
asyncio.run(main())
'''
            worker_python = os.environ.get('COLONYMIND_TEST_ARCHIVE_WORKER_PYTHON', sys.executable)
            worker_process = subprocess.Popen([worker_python, '-I', '-B', '-c', worker_code], env=worker_env,
                                              cwd=tmp_path, stdout=log, stderr=log)
            for _ in range(100):
                if api.get('/v1/host/sessions/archived').status_code == 200:
                    break
                time.sleep(.05)
            else:
                pytest.fail('Hermes heartbeat never enabled the archive capability')
            response = api.post('/v1/host/sessions/side/archive', json=plan['request'])
            assert response.status_code == 200, response.json().get('error')
            receipt = response.json()
            assert receipt['reparented'] == {'cron_jobs': 1}
            assert jobs.get_job(job['id'])['origin'] == MAIN
            assert api.post('/v1/host/sessions/side/archive', json=plan['request']).json() == receipt
            assert api.get('/v1/host/sessions/side/archive').json()['receipt'] == receipt
            assert len(api.get('/v1/host/sessions/archived').json()['items']) == 1
    finally:
        if worker_process is not None:
            worker_process.terminate()
            worker_process.wait(timeout=10)
        process.terminate()
        process.wait(timeout=10)
        sock.close()
        log.close()
