"""Recheck3 sethome race: real persistence/resolver, both delivery lanes."""
import asyncio
from contextlib import closing
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

core, mode, scratch = sys.argv[1:4]
sys.path.insert(0, core)
repo = Path(__file__).resolve().parents[2]
package_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else repo/'plugins/hermes-plugin'
spec = importlib.util.spec_from_file_location('colonymind_hermes', package_dir/'__init__.py', submodule_search_locations=[str(package_dir)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
root = Path(scratch)
root.mkdir(mode=0o700, parents=True, exist_ok=True)
hermes, instance = root/'hermes',root/'instance'
hermes.mkdir(mode=0o700)
instance.mkdir(mode=0o700)
os.environ.update(HERMES_HOME=str(hermes),COLONYMIND_HOME=str(instance),HERMES_DISABLE_LAZY_INSTALLS='1',COLONYMIND_SKIP_DOTENV='1',PYTHON_DOTENV_DISABLED='1')
os.environ.pop('HERMES_KANBAN_TASK',None)
os.environ.pop('TELEGRAM_HOME_CHANNEL',None)
os.environ.pop('TELEGRAM_HOME_CHANNEL_THREAD_ID',None)
os.environ.pop('TELEGRAM_CRON_THREAD_ID',None)
(hermes/'config.yaml').write_text('{}')
(instance/'identity.yaml').write_text(json.dumps({'owner':{'contact_id':'p-01','handles':{'telegram':['1001']}}}))
(instance/'colonymind.yaml').write_text('{}')
(instance/'api.key').write_text('probe-private-key-never-on-argv')
from colonymind_hermes import archive
from colonymind_hermes.client import Settings, ColonyMindClient
from colonymind_hermes.capture import SessionMap
from cron import jobs, scheduler_delivery
settings = Settings('http://unused',instance/'api.key','probe-private-key-never-on-argv',instance,hermes,hermes/'state/outbox.sqlite3')
source = {'platform':'telegram','chat_id':'archive-side','thread_id':None}
main = {'platform':'telegram','chat_id':'archive-main','thread_id':None}
plan = {'operation_id':'op','destination_id':'side','request':{'operation_id':'op','archived':True,'reparent_to_main':True,'main_session_id':'main','source_origin':source,'main_origin':main},'authority':{'speaks_for_owner':True,'sessions:write':True,'owner_id':'p-01'},'created_at':'2026-10-19T12:00:00+00:00','claim':'probe-claim'}
with closing(sqlite3.connect(hermes/'state.db')) as db:
    db.execute('CREATE TABLE gateway_routing (entry_json TEXT)')
    for ident, origin in [('side',source),('main',main)]:
        db.execute('INSERT INTO gateway_routing VALUES (?)',(json.dumps({'session_id':ident,'origin':{**origin,'user_id':'1001','chat_type':'dm'}}),))
    db.commit()
client = ColonyMindClient(settings)
worker = archive.ArchiveWorker(settings,client,SessionMap(settings,client))
from contextlib import contextmanager
from gateway.config import HomeChannel,Platform,persist_home_channel
from colonymind_hermes.archive_core.reparent import _cache_key
stage = mode.split('-')[0] if '-' in mode else 'prep'
failure=mode.endswith('failure')
job=jobs.create_job('concurrent sethome','every 1h',deliver='local' if failure else 'telegram',origin=source)
if failure:
    jobs.update_job(job['id'],{'failure_deliver':'telegram'})
    job=jobs.get_job(job['id'])
before=scheduler_delivery._resolve_delivery_targets(job,for_failure=failure)
key_before=_cache_key(job)
def drift():
    persist_home_channel(HomeChannel(Platform.TELEGRAM,source['chat_id'],'Probe home'))
if stage == 'prep':
    original=archive._checked_jobs_lock
    @contextmanager
    def interleave():
        archive._checked_jobs_lock=original
        drift()
        assert _cache_key(jobs.get_job(job['id']))==key_before
        with original() as fd:yield fd
    archive._checked_jobs_lock=interleave
elif stage == 'commit':
    original=archive.subprocess.Popen
    class AfterCommit(original):
        def communicate(self,*args,**kwargs):
            output=super().communicate(*args,**kwargs)
            archive.subprocess.Popen=original
            drift()
            return output
    archive.subprocess.Popen=AfterCommit
elif stage == 'replay':
    assert worker._apply_bounded(plan)['reparented']=={'cron_jobs':1}
    drift()
else:
    raise ValueError(stage)

try:
    result=worker._apply_bounded(plan)
except archive.ArchiveError as exc:
    assert exc.code == 'busy'
else:
    raise AssertionError('routing drift produced a receipt: ' + json.dumps(result))
after_job=jobs.get_job(job['id'])
after=scheduler_delivery._resolve_delivery_targets(after_job,for_failure=failure)
assert after_job['origin']==source and after[0]['chat_id']==source['chat_id']
with sqlite3.connect(worker.path) as db:
    state,intent,receipt=db.execute('SELECT state,intent_json,receipt_json FROM archive_operations').fetchone()
    assert receipt is None
    assert json.loads(intent)['jobs'][0]['attempted']
    assert db.execute('SELECT count(*) FROM worker_intents').fetchone()[0]==1
    assert db.execute('SELECT count(*) FROM archive_sessions').fetchone()[0]==0
# Retry the durable operation: replay the compensation, never the stale success.
cron_path=jobs._current_cron_store().jobs_file
before_replay=cron_path.read_bytes(),cron_path.stat().st_mtime_ns
result=worker._apply_bounded({**plan,'claim':'retry'})
assert result['error']=='uncertain' and not result['archived'] and not result['tasks_reparented']
assert result['restored']=={'cron_jobs':1} and not result['half_applied']
assert (cron_path.read_bytes(),cron_path.stat().st_mtime_ns)==before_replay
print(json.dumps({'python':sys.executable,'version':sys.version.split()[0],'core':jobs.__file__,
 'adapter':archive.__file__,'gateway_before':before,'origin_after':after_job['origin'],
 'gateway_after':after,'durable_state':state,'retry':result,'positive_receipt':False}),flush=True)
