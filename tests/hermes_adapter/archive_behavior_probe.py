"""Regression probes for archive-core behavior."""
import asyncio
from contextlib import closing
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

core, mode, scratch = sys.argv[1:4]
sys.path.insert(0, core)
repo = Path(__file__).resolve().parents[2]
package_dir = repo/'plugins/hermes-plugin'
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
if mode in ('scope','scope-run'):
    from agent.secret_scope import set_secret_scope,set_multiplex_context
    os.environ['TELEGRAM_HOME_CHANNEL']='host-profile-home'
    mapping = {} if 'missing' in sys.argv else {'TELEGRAM_HOME_CHANNEL': ''}
    set_secret_scope(mapping, profile_home=str(hermes))
    set_multiplex_context(True)
    archive._UNIT_BOOTSTRAP = archive._UNIT_BOOTSTRAP.replace(
        'output.write(json.dumps(response))',
        "from agent.secret_scope import get_secret\nassert get_secret('TELEGRAM_HOME_CHANNEL', 'missing') == %r\noutput.write(json.dumps(response))"
        % ('missing' if 'missing' in sys.argv else ''))
    failure = 'failure' in sys.argv
    job = jobs.create_job('scope test','every 1h',deliver='local' if failure else 'telegram',origin=source)
    if failure:
        jobs.update_job(job['id'], {'failure_deliver': 'telegram'})
        job = jobs.get_job(job['id'])
    before = scheduler_delivery._resolve_delivery_targets(job, for_failure=failure)
    if mode=='scope-run':
        receipts=[]
        class HTTP:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def get(self,*args,**kwargs):
                return SimpleNamespace(status_code=200,json=lambda:{'plan':plan})
            async def post(self,*args,**kwargs):
                receipts.append(kwargs['json']['receipt'])
                worker.close()
                return SimpleNamespace(status_code=200)
        archive.httpx.AsyncClient=HTTP
        async def actual_run():
            worker.task=asyncio.create_task(worker.run())
            await asyncio.gather(worker.task,return_exceptions=True)
        asyncio.run(actual_run())
        assert len(receipts)==1
        result=receipts[0]
    else:
        result = worker._apply_bounded(plan)
    after_job = jobs.get_job(job['id'])
    after = scheduler_delivery._resolve_delivery_targets(after_job, for_failure=failure)
    assert result['reparented'] == {'cron_jobs': 1} and after_job['origin'] == main
    assert before[0]['chat_id'] == source['chat_id'] and after[0]['chat_id'] == main['chat_id']
    print(json.dumps({'gateway_targets_before':before,'receipt':result,'origin_after':after_job['origin'],'gateway_targets_after':after}),flush=True)
elif mode in ('unload','unload-registration'):
    job = jobs.create_job('unload test','every 1h',deliver='origin',origin=source)
    processes=[]
    original = archive.subprocess.Popen
    class Observe(original):
        def __init__(self,*a,**k):
            processes.append({'worker_closed_at_popen':worker.closed,'engine_running':worker.engine_running,'registration_active':getattr(worker.registration,'active',None)})
            super().__init__(*a,**k)
    archive.subprocess.Popen=Observe
    class HTTP:
        def __init__(self,**kwargs): pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,*args,**kwargs):
            if mode=='unload-registration':
                worker.registration.dispose()
            else:
                worker.close()
            assert worker.closed
            return SimpleNamespace(status_code=200,json=lambda:{'plan':plan})
        async def post(self,*args,**kwargs):
            raise AssertionError('closed worker submitted receipt')
    archive.httpx.AsyncClient=HTTP
    async def run():
        if mode=='unload-registration':
            from hermes_cli.plugins import PluginContext,PluginManager,PluginManifest
            ctx=PluginContext(PluginManifest(name='real-unload'),PluginManager())
            worker.start(ctx)
        else:
            worker.task=asyncio.create_task(worker.run())
        await asyncio.gather(worker.task,return_exceptions=True)
        deadline=time.monotonic()+8
        while worker.engine_running and time.monotonic()<deadline:
            await asyncio.sleep(.01)
        assert not worker.engine_running
    asyncio.run(run())
    assert not processes and jobs.get_job(job['id'])['origin'] == source
    print(json.dumps({'launches':processes,'worker_closed':worker.closed,'origin_after':jobs.get_job(job['id'])['origin']}),flush=True)
elif mode in ('park','park-rollback'):
    job=jobs.create_job('actual supervised unit','every 1h',deliver='origin',origin=source)
    if mode=='park-rollback':
        jobs.create_job('second compensation target','every 1h',deliver='origin',origin=source)
    worker.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    worker.lease=os.open(str(worker.path)+'.worker.lock',os.O_CREAT|os.O_RDWR,0o600)
    fcntl.flock(worker.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'plan.json').write_text(json.dumps(plan))
    if mode.startswith('park'):
        phase='commit' if mode=='park' else 'rollback'
        injection='''
from _colonymind_archive_unit import archive as unit
from pathlib import Path
import os, threading, json, sys
original=unit.archive_reparent
def park(**kwargs):
    if kwargs['phase']==%r:
        Path(%r).write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid()}))
        threading.Event().wait()
    if %r and kwargs['phase']=='commit':
        raise RuntimeError('trigger compensation')
    return original(**kwargs)
unit.archive_reparent=park
''' % (phase,str(root/'parked.json'),mode=='park-rollback')
        if mode=='park-rollback':
            injection='''
from _colonymind_archive_unit import archive as unit
from cron import jobs
from pathlib import Path
import os,threading,json
engine=unit.archive_reparent
def fail_commit(**kwargs):
    if kwargs['phase']=='commit':
        raise RuntimeError('trigger compensation')
    return engine(**kwargs)
unit.archive_reparent=fail_commit
update=jobs.update_job
def park_after_restore(ident,changes):
    result=update(ident,changes)
    if changes.get('origin',{}).get('chat_id')=='archive-side':
        Path(%r).write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid()}))
        threading.Event().wait()
    return result
jobs.update_job=park_after_restore
''' % str(root/'parked.json')
        archive._UNIT_BOOTSTRAP=archive._UNIT_BOOTSTRAP.replace('try:\n    response',injection+'\ntry:\n    response')
    async def run():
        work=worker._engine_unit(plan)
        if mode.startswith('park'):
            while not (root/'parked.json').exists():
                if work.done():
                    await work
                    raise AssertionError('unit ended without marker')
                await asyncio.sleep(.01)
            print('parked '+(root/'parked.json').read_text(),flush=True)
        try:
            result=await work
            print(json.dumps({'result':result}),flush=True)
        except archive.ArchiveError as exc:
            print(json.dumps({'failure':exc.body}),flush=True)
        finally:
            worker.close()
            worker._release_lease()
    asyncio.run(run())
elif mode=='discovery':
    plugin=hermes/'plugins/b01-sentinel'
    plugin.mkdir(mode=0o700,parents=True)
    marker=root/'plugin-registrations.jsonl'
    (plugin/'plugin.yaml').write_text('name: b01-sentinel\nversion: 1.0.0\nentry_point: __init__.py\n')
    code="from pathlib import Path\nimport json,os,threading\ndef register(ctx):\n    with Path(%r).open('a') as fd:fd.write(json.dumps({'pid':os.getpid(),'thread':threading.current_thread().name})+'\\n')\n    prefix = 'native-'\n    def parse(ref):\n        return (ref[len(prefix):], None) if ref.startswith(prefix) else None\n    ctx.register_platform('archivecustom', 'Archive custom', lambda cfg: None, lambda: True, cron_deliver_env_var='ARCHIVECUSTOM_HOME', parse_target_ref_fn=parse, validate_target_ref_fn=lambda ref: ref == 'custom-main')\n" % str(marker)
    (plugin/'__init__.py').write_text(code)
    (hermes/'config.yaml').write_text(json.dumps({'plugins':{'enabled':['b01-sentinel']}}))
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    before=[json.loads(x) for x in marker.read_text().splitlines()]
    jobs.create_job('explicit target triggers platform prep','every 1h',deliver='telegram:archive-side',origin=source)
    custom = jobs.create_job('custom closure parser', 'every 1h', deliver='archivecustom:native-custom-main', origin=source)
    jobs.update_job(custom['id'], {'failure_deliver': 'origin'})
    result=worker._apply_bounded(plan)
    after=[json.loads(x) for x in marker.read_text().splitlines()]
    assert after == before and len(before) == 1
    assert result['reparented'] == {'cron_jobs': 1}
    assert jobs.get_job(custom['id'])['origin'] == main
    print(json.dumps({'registrations_before':before,'registrations_after':after,'result':result}),flush=True)
else:
    raise ValueError(mode)
