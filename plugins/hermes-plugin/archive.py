"""Authenticated pull worker; bounded cron units run in the gateway's interpreter.

No inbound socket or credential is added. The existing adapter credential polls
the sidecar, and the adapter's private journal replays receipts after lost replies.
"""
from __future__ import annotations

import asyncio
import atexit
from contextlib import closing, contextmanager
import contextvars
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import quote
import weakref

import httpx

from .archive_core.reparent import archive_reparent, _cache_key, _core, _resolver, _verify_delivery, _with_origin
from .archive_core.store import ArchiveError, ArchiveStore, _json, _parents

PLANS = '/v1/host/sessions/archive/plans'
_workers = weakref.WeakSet()
UNIT_SECONDS = 5

# Use a fresh interpreter, never fork a threaded gateway's Python state. The
# synthetic package also supports stock Hermes' directory-loaded adapter.
_UNIT_BOOTSTRAP = '''
import os, signal, sys, time
# Survive loss of the daemon supervisor: the child owns the same absolute
# deadline and the inherited worker lease until its last possible mutation.
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.setitimer(signal.ITIMER_REAL, max(.001, float(sys.argv[3]) - time.monotonic()))
# Attach death handling before imports or payload reads. Linux supplies a
# kernel death signal; macOS supplies a process-exit event. Recheck the parent
# after subscribing to cover death between exec and subscription.
parent = int(sys.argv[4])
if sys.platform == 'linux':
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0):
        raise OSError('parent-death signal unavailable')
elif sys.platform == 'darwin':
    import select, threading
    watcher = select.kqueue()
    watcher.control([select.kevent(parent, filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD, fflags=select.KQ_NOTE_EXIT)], 0, 0)
    def parent_exit():
        watcher.control(None, 1)
        os._exit(1)
    threading.Thread(target=parent_exit, daemon=True).start()
else:
    raise RuntimeError('parent-death handling unavailable')
if os.getppid() != parent:
    os._exit(1)
import json, sys, types
payload = json.load(sys.stdin)
sys.path[:] = payload['sys_path']
from agent import secret_scope
scope = payload['scope']
secret_scope.set_secret_scope(scope['mapping'], profile_home=scope['home'])
secret_scope.set_multiplex_context(scope['multiplex'])
from hermes_cli.plugins import get_plugin_manager
get_plugin_manager()._discovered = True
output, sys.stdout = sys.stdout, sys.stderr
package = types.ModuleType('_colonymind_archive_unit')
package.__path__ = [sys.argv[1]]
sys.modules[package.__name__] = package
from _colonymind_archive_unit.archive import _child_main
from _colonymind_archive_unit.archive_core.store import ArchiveError
try:
    response = {'result': _child_main(payload, int(sys.argv[2]))}
except ArchiveError as exc:
    response = {'failure': exc.body}
output.write(json.dumps(response))
output.flush()
'''


def _after_fork():
    for worker in tuple(_workers):
        worker.pid = -1
        if worker.lease is not None:
            os.close(worker.lease)
            worker.lease = None


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _identity(origin):
    return (str(origin.get('platform', '')).lower(), origin.get('chat_id'), origin.get('thread_id'))


@contextmanager
def _checked_jobs_lock():
    """Checked integration with Hermes' reentrant cron lock (both pinned cores).

    Order is cron RLock, cron flock, then journal lock. Establish the outer
    section ourselves: stock _jobs_lock deliberately fails open on flock errors.
    Its nested-call protocol makes every ordinary core writer reuse this checked
    section, including updates made by compensation. Never borrow an unchecked
    outer section, and never wait on a competing writer.
    """
    from cron import jobs
    if jobs.fcntl is not fcntl:
        raise ArchiveError('busy')
    if not jobs._jobs_file_lock.acquire(blocking=False):
        raise ArchiveError('busy')
    lock = None
    entered = False
    try:
        if getattr(jobs._jobs_lock_state, 'depth', 0):
            raise ArchiveError('busy')
        try:
            jobs.ensure_dirs()
            lock = open(jobs._jobs_lock_file(), 'a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ArchiveError('busy') from None
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.load_stamp = None
        entered = True
        with jobs._jobs_lock():
            yield lock.fileno()
    finally:
        if entered:
            jobs._jobs_lock_state.depth = 0
            jobs._jobs_lock_state.load_stamp = None
        if lock is not None:
            lock.close()
        jobs._jobs_file_lock.release()


def _child_main(payload, lock_fd):
    """Borrow the parent's checked flock; only the parent releases exclusion.

    No child thread/executor may outlive this synchronous unit. On timeout the
    parent kills and reaps this process before releasing its RLock/flock. Cron's
    atomic replace and the journal's pre-write intent make termination equivalent
    to the already supported crash during an update, including compensation.
    """
    from cron import jobs
    from .client import Settings, ColonyMindClient
    from .capture import SessionMap
    from .archive_core import reparent
    # All delivery decisions were made by the gateway's existing plugins and
    # scoped resolver, before taking any cron lock. Never discover/start plugins
    # in this transient process. An unseen job/changed lane fails closed.
    def targets(job):
        try:
            result = payload['targets'][_cache_key(job)]
        except KeyError:
            raise ArchiveError('busy') from None
        if isinstance(result, dict):
            raise ArchiveError(result['error'], **{k: v for k, v in result.items() if k != 'error'})
        return result
    reparent._resolver = lambda: targets
    fields = payload['settings']
    for key in ('key_file', 'home', 'hermes_home', 'outbox_path', 'body_ledger_path'):
        if fields[key] is not None:
            fields[key] = Path(fields[key])
    settings = Settings(**fields)
    store = jobs._CronStorePaths(**{key: Path(value) for key, value in payload['cron_store'].items()})
    token = jobs._cron_store_override.set(store)
    try:
        # Check the passed open-file description belongs to this exact store.
        held, expected = os.fstat(lock_fd), os.stat(jobs._jobs_lock_file())
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise ArchiveError('busy')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with jobs._jobs_file_lock:
            jobs._jobs_lock_state.depth = 1
            jobs._jobs_lock_state.load_stamp = None
            try:
                client = ColonyMindClient(settings)
                worker = ArchiveWorker(settings, client, SessionMap(settings, client))
                result = worker._apply_locked(payload['plan'], rollback=payload.get('rollback', False))
                intent = snapshot = None
                if not payload.get('rollback') and 'error' not in result:
                    snapshot = jobs.list_jobs(include_disabled=True)
                    with closing(sqlite3.connect(worker.path.as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
                        intent = json.loads(db.execute(
                            'SELECT intent_json FROM archive_operations WHERE operation_id=?',
                            (payload['plan']['operation_id'],)).fetchone()[0])
            finally:
                jobs._jobs_lock_state.depth = 0
                jobs._jobs_lock_state.load_stamp = None
        return result, intent, snapshot
    finally:
        jobs._cron_store_override.reset(token)


def _verify_grant(plan, settings):
    authority = plan.get('authority', {})
    if (authority.get('speaks_for_owner') is not True or authority.get('sessions:write') is not True
            or not settings.owner_contact_id() or authority.get('owner_id') != settings.owner_contact_id()):
        raise ArchiveError('unauthorized', refused=[plan['destination_id']])


def verify_owner(plan, settings, sessions):
    """Recheck the sidecar grant AND Hermes' own session/origin/sender binding.

    Read state.db in mode=ro, without constructing SessionDB (which migrates).
    Unknown, ambiguous, missing or mismatched bindings confer no write authority.
    """
    _verify_grant(plan, settings)
    try:
        path = (settings.hermes_home / 'state.db').resolve()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            entries = [json.loads(row[0]) for row in db.execute('SELECT entry_json FROM gateway_routing')]
        for ident, origin in ((plan['destination_id'], plan['request']['source_origin']),
                              (plan['request']['main_session_id'], plan['request']['main_origin'])):
            found = [entry for entry in entries if ident in (entry.get('session_id'), entry.get('session_key'))]
            if not found:
                raise ArchiveError('unauthorized', refused=[ident])
            for entry in found:
                bound = entry.get('origin') or {}
                sender = bound.get('user_id')
                if (_identity(bound) != _identity(origin) or not sender
                        or bound.get('is_bot') is True
                        or not sessions.sender_is_owner(str(bound.get('platform', '')), str(sender))):
                    raise ArchiveError('unauthorized', refused=[ident])
    except ArchiveError:
        raise
    except Exception:
        raise ArchiveError('source_unavailable') from None


class ArchiveWorker:
    def __init__(self, settings, client, sessions):
        self.settings, self.client, self.sessions = settings, client, sessions
        self.path = settings.outbox_path.with_name(settings.outbox_path.stem + '-archive.sqlite3')
        self.pid = os.getpid()
        self.closed = False
        self.task = None
        self.lease = None
        self.thread = None
        self.emitter = None
        self.lifecycle = threading.RLock()
        self.registration = None
        self.engine_running = False
        _workers.add(self)

    def apply(self, plan):
        with _checked_jobs_lock():
            return self._apply_locked(plan)

    def _apply_bounded(self, plan, *, rollback=False):
        """Hold gateway exclusion until a finite child exits, including on error.

        A timed-out thread cannot safely give up the lock: it could still write.
        Reaping a killed child provides that guarantee even for a callback stuck
        in a blocking C call. An interrupted intent stays pending for recovery;
        no transport refusal/success is manufactured from the deadline.
        """
        from cron import jobs
        from agent import secret_scope
        # Resolver/plugin callbacks can stall: run them in the daemon gateway
        # thread before exclusion. Capture current and intended-origin views,
        # including custom parsers and both delivery lanes, without copying code.
        targets = {}
        if not rollback:
            resolve = _resolver()
            for job in jobs.list_jobs(include_disabled=True):
                for view in (job, _with_origin(job, plan['request']['main_origin'])):
                    try:
                        targets[_cache_key(view)] = resolve(view)
                    except ArchiveError as exc:
                        targets[_cache_key(view)] = exc.body
        scope = secret_scope.current_secret_scope()
        deadline = time.monotonic() + UNIT_SECONDS
        with _checked_jobs_lock() as lock_fd:
            payload = json.dumps({
                'plan': plan, 'settings': asdict(self.settings),
                'sys_path': [os.path.abspath(path) for path in sys.path],
                'cron_store': asdict(jobs._current_cron_store()),
                'targets': targets, 'rollback': rollback,
                'scope': {'mapping': None if scope is None else dict(scope),
                          'home': secret_scope.current_secret_scope_home(),
                          'multiplex': secret_scope.is_multiplex_active()
                          or (scope is not None and secret_scope.serves_routed_profile())},
            }, default=str)
            env = dict(os.environ, HERMES_HOME=str(self.settings.hermes_home),
                       COLONYMIND_HOME=str(self.settings.home))
            inherited = (lock_fd,) if self.lease is None else (lock_fd, self.lease)
            with self.lifecycle:
                self._admit()
                process = subprocess.Popen([sys.executable, '-I', '-B', '-c', _UNIT_BOOTSTRAP,
                                           str(Path(__file__).resolve().parent), str(lock_fd),
                                           str(deadline), str(self.pid)],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, pass_fds=inherited,
                                          env=env, text=True)
            with process:
                try:
                    output, _ = process.communicate(payload, timeout=max(0, deadline - time.monotonic()))
                    if process.returncode:
                        raise ArchiveError('busy')
                    response = json.loads(output)
                    if 'failure' in response:
                        failure = response['failure']
                        raise ArchiveError(failure['error'], **{k: v for k, v in failure.items() if k != 'error'})
                    result, intent, snapshot = response['result']
                except subprocess.TimeoutExpired:
                    raise ArchiveError('busy') from None
                finally:
                    # Do not unlock or release the reload lease until mutation is
                    # impossible. communicate() also waits on a normal completion.
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        if not rollback and 'error' not in result:
            try:
                # Fresh gateway/plugin/profile answers AFTER commit, also on
                # receipt replay. Callbacks remain outside cron exclusion.
                _verify_delivery(plan['request'], intent, snapshot)
            except Exception:
                self._apply_bounded(plan, rollback=True)
                raise ArchiveError('busy') from None
        return result

    def _apply_locked(self, plan, *, rollback=False):
        journal = ArchiveStore(self.path, stamp=lambda: plan['created_at'])
        with journal._writer() as db:
            db.execute('CREATE TABLE IF NOT EXISTS worker_results '
                       '(operation_id TEXT PRIMARY KEY, plan_json TEXT NOT NULL, result_json TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS worker_intents '
                       '(operation_id TEXT PRIMARY KEY, plan_json TEXT NOT NULL)')
            stable = _json({k: v for k, v in plan.items() if k != 'claim'})
            authority = db.execute('SELECT plan_json FROM worker_intents WHERE operation_id=?',
                                   (plan['operation_id'],)).fetchone()
            if authority and authority[0] != stable:
                raise ArchiveError('idempotency_conflict')
            if rollback:
                # Revoke the unsent receipt before the existing recovery path.
                # Keep the durable authority/intent, including attempted flags.
                db.execute("UPDATE archive_operations SET state='rolling_back', receipt_json=NULL "
                           'WHERE operation_id=?', (plan['operation_id'],))
                db.execute('DELETE FROM worker_results WHERE operation_id=?', (plan['operation_id'],))
                db.execute('DELETE FROM archive_sessions WHERE destination_id=?', (plan['destination_id'],))
            old = db.execute('SELECT plan_json,result_json FROM worker_results WHERE operation_id=?',
                             (plan['operation_id'],)).fetchone()
            if old:
                if old[0] != stable:
                    raise ArchiveError('idempotency_conflict')
                result = json.loads(old[1])
                if not (result.get('half_applied') or result.get('receipt_incomplete')):
                    return result
            applied = db.execute('SELECT destination_id,request_json,receipt_json FROM archive_operations '
                                 'WHERE operation_id=?', (plan['operation_id'],)).fetchone()
            if applied:
                if applied[0] != plan['destination_id'] or applied[1] != _json(plan['request']):
                    raise ArchiveError('idempotency_conflict')
                if applied[2]:
                    # The process may have died between the engine's receipt
                    # commit and our transport-result commit. Replay before any
                    # ownership recheck: this path performs no cron write.
                    db.execute('INSERT INTO worker_results VALUES (?,?,?)',
                               (plan['operation_id'], stable, applied[2]))
                    db.commit()
                    return json.loads(applied[2])
            # Carry the pre-upgrade overlay state into the journal, so an
            # unarchive refusal still reports archived=true after deployment.
            previous = plan.get('previous_receipt')
            if previous:
                db.execute('INSERT OR IGNORE INTO archive_sessions VALUES (?,?,?,?,?)',
                           (plan['destination_id'], previous['archived'], previous['revision'],
                            previous['archived_at'], _json(previous)))
            db.commit()
        def authorize(db):
            verify_owner(plan, self.settings, self.sessions)
            db.execute('INSERT OR IGNORE INTO worker_intents VALUES (?,?)', (plan['operation_id'], stable))
            db.commit()

        def engine(**kwargs):
            if kwargs['phase'] == 'rollback':
                # Recovery uses the authority durably accepted before the write,
                # independent of subsequently missing/compressed session bindings
                # or a transient sender resolver. Core rollback verifies origins.
                with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
                    row = db.execute('SELECT plan_json FROM worker_intents WHERE operation_id=?',
                                     (kwargs['request']['operation_id'],)).fetchone()
                if row is None:
                    raise ArchiveError('unauthorized')
                recorded = json.loads(row[0])
                if (recorded['destination_id'] != kwargs['session_id']
                        or recorded['request'] != kwargs['request']):
                    raise ArchiveError('unauthorized')
                _verify_grant(recorded, self.settings)
            return archive_reparent(**kwargs)

        try:
            # archive recovers all pending intents BEFORE invoking authorize for
            # a new operation, and never reauthorizes an interrupted operation.
            result = journal.archive(plan['destination_id'], plan['request'], engine, authorize=authorize)
        except ArchiveError as exc:
            if exc.code == 'busy':
                raise  # No safe recovery result yet; leave the remote plan pending.
            result = exc.body
        # If this write fails after cron succeeded, journal.archive already has
        # the durable receipt. Reclaiming returns it without another mutation.
        with journal._writer() as db:
            db.execute('INSERT INTO worker_results VALUES (?,?,?) ON CONFLICT(operation_id) '
                       'DO UPDATE SET result_json=excluded.result_json',
                       (plan['operation_id'], stable, _json(result)))
            db.commit()
        return result

    def _engine_unit(self, plan):
        """Cancellation is prompt; the supervisor retains the lease until reaping."""
        loop = asyncio.get_running_loop()
        completed = loop.create_future()

        def deliver(result, failure):
            if not completed.done():
                if failure is not None:
                    completed.set_exception(failure)
                else:
                    completed.set_result(result)

        def execute():
            result, failure = None, None
            try:
                result = self._apply_bounded(plan)
            except BaseException as exc:
                failure = exc
            finally:
                with self.lifecycle:
                    self.engine_running = False
                    if self.closed:
                        self._release_lease()
            try:
                loop.call_soon_threadsafe(deliver, result, failure)
            except RuntimeError:
                pass  # Process exit leaves the durable intent for next start.

        with self.lifecycle:
            self._admit()
            self.engine_running = True
            try:
                threading.Thread(target=contextvars.copy_context().run, args=(execute,),
                                 name='colonymind-session-archive-engine', daemon=True).start()
            except BaseException:
                self.engine_running = False
                raise
        return completed

    def _admit(self):
        # Called under lifecycle both at dispatch and immediately before spawn.
        if (self.closed or self.pid != os.getpid()
                or (self.registration is not None and not self.registration.active)):
            raise ArchiveError('busy')

    async def run(self):
        try:
            _parents(self.path.parent)
            self.lease = os.open(str(self.path) + '.worker.lock', os.O_CREAT | os.O_RDWR, 0o600)
            os.fchmod(self.lease, 0o600)
            while not self.closed and self.pid == os.getpid():
                try:
                    fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(5)
            if self.closed or self.pid != os.getpid():
                return
            async with httpx.AsyncClient(base_url=self.client.url, headers=self.client.headers(),
                                         timeout=30, trust_env=False) as http:
                while not self.closed and self.pid == os.getpid():
                    try:
                        # A successful heartbeat is evidence that both core APIs
                        # are present in this interpreter. Probe without job I/O.
                        if not all(callable(api) for api in _core()):
                            raise ArchiveError('source_unavailable')
                        _resolver()
                        response = await http.get(PLANS, params={'wait': 25})
                        if response.status_code != 200:
                            # Older sidecars return 404: no log spam, bounded retry.
                            await asyncio.sleep(30)
                            continue
                        plan = response.json().get('plan')
                        if plan is None:
                            continue
                        # Only apply's durable outcomes are submitted. Exceptions
                        # (including busy) leave the plan pending for recovery.
                        work = self._engine_unit(plan)
                        try:
                            result = await asyncio.shield(work)
                        except asyncio.CancelledError:
                            # The daemon supervisor retains the worker lease.
                            # Cancellation must let asyncio.run shut down;
                            # reload still cannot start a competing unit.
                            work.cancel()
                            raise
                        token = quote(plan['operation_id'], safe='').replace('.', '%2E')
                        endpoint = PLANS + '/' + token + '/receipt'
                        receipt = await http.post(endpoint, json={'claim': plan['claim'], 'receipt': result})
                        if receipt.status_code != 200:
                            await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Leave the lease to expire. Recovery uses the journal,
                        # never a guessed success or an in-memory applied flag.
                        await asyncio.sleep(5)
        finally:
            with self.lifecycle:
                self.closed = True
                if not self.engine_running:
                    self._release_lease()

    def _release_lease(self):
        if self.lease is not None:
            os.close(self.lease)
            self.lease = None

    def close(self):
        with self.lifecycle:
            self._close_locked()

    def _close_locked(self):
        self.closed = True
        if self.pid != os.getpid():
            return
        if self.emitter is not None:
            self.emitter.unsubscribe(self._gateway_events)
            self.emitter = None
        if self.task is not None:
            loop = self.task.get_loop()
            if not loop.is_closed():
                loop.call_soon_threadsafe(self.task.cancel)
        atexit.unregister(self.close)

    def _gateway_events(self, batch):
        if any(event.get('event') == 'gateway_health' and event.get('name') == 'gateway.exit'
               and event.get('pid') == self.pid for event in batch):
            self.close()

    def start(self, ctx):
        with self.lifecycle:
            self._start_locked(ctx)

    def _start_locked(self, ctx):
        # Stock Hermes normally calls register() on its deadline loader thread,
        # without an event loop. Host one loop like Body/Activity host workers;
        # spawn_task still owns cancellation in the plugin's registration ledger.
        if os.environ.get('HERMES_KANBAN_TASK'):
            return
        if not callable(getattr(ctx, 'spawn_task', None)) or not callable(getattr(ctx, 'on_unload', None)):
            return
        self.registration = ctx.on_unload(self.close)
        if self.closed or self.registration is None or not self.registration.active:
            return
        # Initialize the core package before a background poll can race the
        # registering caller's own cron imports. This is import-only, never an
        # engine unit. Let deadline cleanup dispose our lease during the probe.
        core_ready = False
        self.lifecycle.release()
        try:
            if not all(callable(api) for api in _core()):
                raise ArchiveError('source_unavailable')
            _resolver()
            core_ready = True
        except Exception:
            pass
        finally:
            self.lifecycle.acquire()
        if not core_ready or self.closed or not self.registration.active:
            self._close_locked()
            return
        atexit.register(self.close)
        try:
            from agent.monitoring.emitter import get_emitter
            self.emitter = get_emitter()
            self.emitter.subscribe(self._gateway_events)
        except ImportError:
            pass
        if self.closed or not self.registration.active:
            self._close_locked()
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            ready = threading.Event()
            async def supervised():
                with self.lifecycle:
                    if self.closed or not self.registration.active:
                        ready.set()
                        return
                    self.task = ctx.spawn_task(self.run(), name='colonymind-session-archive')
                ready.set()
                if self.closed:
                    self.task.cancel()
                await self.task
            def run_loop():
                try:
                    asyncio.run(supervised())
                except asyncio.CancelledError:
                    pass
                finally:
                    ready.set()
            self.thread = threading.Thread(target=contextvars.copy_context().run, args=(run_loop,),
                                           name='colonymind-session-archive', daemon=True)
            self.thread.start()
            # Let the daemon acquire lifecycle before the startup rendezvous.
            self.lifecycle.release()
            try:
                started = ready.wait(1)
            finally:
                self.lifecycle.acquire()
            if not started:
                self.close()
        else:
            self.task = ctx.spawn_task(self.run(), name='colonymind-session-archive')
