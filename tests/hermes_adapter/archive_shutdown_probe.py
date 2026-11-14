"""Disposable child for the archive/tick/interpreter-exit regression.

Real tick, scan/advance, pool, submit guard, claim and lock acquisition. Only
unrelated housekeeping, network transport and job execution are substituted.
The engine parks *after* its real cron update, in the commit callback.
"""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace


package_dir, plan_file, version = sys.argv[1:4]
assert sys.version.startswith(version), sys.version
spec = importlib.util.spec_from_file_location(
    'colonymind_hermes', Path(package_dir) / '__init__.py', submodule_search_locations=[package_dir])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
from colonymind_hermes import archive
from colonymind_hermes.capture import SessionMap
from colonymind_hermes.client import ColonyMindClient, load_settings
from cron import jobs, scheduler
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_loader import run_with_load_deadline

plan = json.loads(Path(plan_file).read_text())
settings = load_settings()
client = ColonyMindClient(settings)
worker = archive.ArchiveWorker(settings, client, SessionMap(settings, client))
marker = Path(plan_file).with_suffix('.parked')

# Inject into the actual fresh unit, without a production environment switch.
injection = '''
from _colonymind_archive_unit import archive as unit
from pathlib import Path
import os, threading
original = unit.archive_reparent
def park(**kwargs):
    if kwargs['phase'] == 'commit':
        Path(%r).write_text(str(os.getpid()))
        threading.Event().wait()
    return original(**kwargs)
unit.archive_reparent = park
''' % str(marker)
archive._UNIT_BOOTSTRAP = getattr(archive, '_UNIT_BOOTSTRAP', '').replace(
    'try:\n    response', injection + '\ntry:\n    response')
if '--legacy' in sys.argv:
    # Run this same interleaving against the preceding wheel as a negative
    # control. Its engine lives in the gateway's daemon thread, without a child.
    original_engine = archive.archive_reparent
    def legacy_park(**kwargs):
        if kwargs['phase'] == 'commit':
            marker.write_text(str(os.getpid()))
            threading.Event().wait()
        return original_engine(**kwargs)
    archive.archive_reparent = legacy_park

class HTTP:
    def __init__(self, **kwargs):
        self.offered = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, *args, **kwargs):
        if self.offered:
            await asyncio.sleep(30)
        self.offered = True
        return SimpleNamespace(status_code=200, json=lambda: {'plan': plan})

    async def post(self, *args, **kwargs):
        raise AssertionError('an interrupted unit must leave the remote plan pending')

archive.httpx.AsyncClient = HTTP
ctx = PluginContext(PluginManifest(name='archive-shutdown'), PluginManager())
target = jobs.create_job('archive target', 'every 1h', deliver='origin', origin=plan['request']['source_origin'])
due = jobs.create_job('tick target', 'every 1h', deliver='local', origin=plan['request']['main_origin'])
jobs.update_job(due['id'], {'next_run_at': (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})

entered, claimed = threading.Event(), threading.Event()
original_with_job = jobs._with_job
def with_job(*args, **kwargs):
    if threading.current_thread().name.startswith('cron-parallel'):
        entered.set()
    return original_with_job(*args, **kwargs)
jobs._with_job = with_job
def run_job(*args, **kwargs):
    try:
        os.kill(int(marker.read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('cron lock released before the engine was reaped')
    claimed.set()
    print('claim drained after reap', flush=True)
    return True
scheduler.run_one_job = run_job

# None of these substitutes handles scan, advance, submit, claim or cron locks.
scheduler._should_yield_tick_to_fresh_gateway = lambda: None
scheduler._maybe_reap_dead_owners = lambda: None
scheduler._maybe_run_worktree_maintenance = lambda: None
scheduler._sweep_stale_inflight_for_tick = lambda due: None
scheduler._sweep_mcp_orphans = lambda: None
# The tick's Bot Chat drain, in the module this source keeps it in.
bot_chat_drains = [(module, name) for module, name in (
    ('cron.bot_chat_delivery', 'drain_in_background'), ('cron.bot_chat_legacy', 'drain_legacy_pending'))
    if importlib.util.find_spec(module) is not None]
assert bot_chat_drains, 'this Hermes source keeps the tick\'s Bot Chat drain in a module the probe does not know'
for module, name in bot_chat_drains:
    module = importlib.import_module(module)
    assert callable(getattr(module, name)), (module.__name__, name)
    setattr(module, name, lambda: None)

def interleave():
    # _resolve_max_parallel_workers is called AFTER the real scan and advance.
    run_with_load_deadline('archive-shutdown', ctx, lambda: worker.start(ctx))
    deadline = time.monotonic() + 4
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert marker.exists(), 'engine did not reach commit before its deadline'
    assert not jobs._jobs_file_lock.acquire(blocking=False), 'gateway RLock not held'
    for path in (jobs._jobs_lock_file(), Path(str(worker.path) + '.lock')):
        with path.open('rb') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('archive flock not held: ' + str(path))
    # Read the real pre-write intent while the child's commit transaction waits.
    with closing(sqlite3.connect(worker.path, timeout=0)) as db:
        row = db.execute('SELECT state,intent_json,receipt_json FROM archive_operations').fetchone()
    assert row[0] == 'applying' and row[2] is None
    assert json.loads(row[1])['jobs'][0]['attempted']
    assert next(j for j in jobs.load_jobs() if j['id'] == target['id'])['origin'] == plan['request']['main_origin']
    return 1

scheduler._resolve_max_parallel_workers = interleave
assert scheduler.tick(verbose=False, sync=False) == 1
assert entered.wait(1), 'ordinary scheduler claim never reached _with_job'
assert not claimed.is_set(), 'claim escaped archive exclusion'
worker.close()
assert worker.closed and worker.engine_running and worker.lease is not None
with open(str(worker.path) + '.worker.lock', 'rb') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise AssertionError('reload lease released while child can still mutate')
print('shutdown reached: closed=True engine_running=True; core=' + jobs.__file__, flush=True)
# Return with a real cron-parallel executor waiting on the RLock. Python's own
# executor join must finish after the supervisor kills/reaps the parked engine.
