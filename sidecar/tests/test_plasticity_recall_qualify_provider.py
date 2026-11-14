"""Socket-bound host prefetch, live query evidence and erasure of a trained turn."""
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import socket
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from colonymind.plasticity import _io
from colonymind.plasticity.nightly import Runtime
from colonymind.plasticity.recall_qualify import qualify
from recall_route import OWNER, recall_app
from test_plasticity_query_adapter_driver import Config
from test_plasticity_recall_qualify import FixtureScorer, queries


@pytest.mark.asyncio
@pytest.mark.parametrize('socket_bound', [False, True], ids=['in-process', 'socket-bound'])
async def test_socket_owner_prefetch_logs_once_then_erasure_unserves(tmp_path, monkeypatch, socket_bound):
    # Fail explicitly when a sandbox denies sockets, for the driver's unsandboxed rerun.
    listener = None
    if socket_bound:
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        listener.setblocking(False)
    server = task = active = app = client = None
    try:
        app = await asyncio.wait_for(recall_app(monkeypatch, tmp_path / 'instance', initiatives=0), 15)
        config = Config(tmp_path / 'plasticity')
        config.values['plasticity.learners']['recall-test'].update(dims=3,
            document_identity=asdict(app.embedder.index_identity.document_identity))
        now = datetime.now(timezone.utc)
        active = Runtime(config, clock=lambda: now)
        driver = active.loaded['recall-test'].driver
        driver.attach(active)
        monkeypatch.setattr('colonymind.api.routers.mind._mind',
                            SimpleNamespace(plasticity=active, enabled=False))
        if listener is not None:
            server = uvicorn.Server(uvicorn.Config(app.app, log_level='error', lifespan='off'))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(.01)
            assert server.started
        path = Path(__file__).resolve().parents[2] / 'plugins/colonymind-memory/provider.py'
        spec = importlib.util.spec_from_file_location('turn_provider_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not socket_bound:
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://fixture')
            loop = asyncio.get_running_loop()
            def request(timeout, method, url, **kwargs):
                return asyncio.run_coroutine_threadsafe(client.request(method, url, **kwargs), loop).result(timeout)
            monkeypatch.setattr(module, '_request_within', request)
        from onekey import KEY
        key = tmp_path / 'api.key'
        key.write_text(KEY)
        url = f'http://127.0.0.1:{listener.getsockname()[1]}' if listener is not None else 'http://fixture'
        provider = module.ColonyMindMemoryProvider({'url': url,
                                                   'key_file': str(key), 'contact_id': OWNER})
        provider.initialize('owner-session', platform='cli')
        provider.resolve_contact('cli', '', 'owner-session', turn_id='owner-recall')
        for _ in range(2):
            sections = await asyncio.to_thread(provider._prefetch_sections, 'hydrofoil departure',
                                               session_id='owner-session', contact_id=OWNER)
            assert sections
        # The response precedes accepted context writes. Finish both requests'
        # trace/selection writes before this fixture edits their training evidence.
        from colonymind.memory import turn_budget
        loop = asyncio.get_running_loop()
        pending = [task for task in turn_budget._PENDING if task.get_loop() is loop]
        await asyncio.wait_for(asyncio.gather(*pending), 15)
        assert driver.queries.get('owner-recall') is not None
        with driver.queries._transaction() as conn:
            assert conn.execute('SELECT count(*) FROM query_vectors').fetchone()[0] == 1
        assert len(active.loaded['recall-test'].comparator.records()) == 1
        record = driver.evidence('owner-recall')
        positive = next(p['id'] for p in record['passages'] if p['source_id'] == 'answer')
        negative = next(p['id'] for p in record['passages'] if p['source_id'] != 'answer')
        record['at'] = (now - timedelta(days=5)).isoformat()
        record['shown'] = {positive: 1, negative: 2}
        _io.write_json(driver._path('owner-recall'), record)
        driver.passage_reader = lambda p: [0., 1., 0.] if p['id'] == positive else [1., 0., 0.]
        assert driver.signal('owner-recall', preferred=positive, disfavoured=negative,
                             teacher='owner_label', evidence_refs=['owner-verdict'])
        await asyncio.to_thread(driver.period)
        await qualify(driver, queries(), scorer=FixtureScorer(driver))
        assert (await asyncio.to_thread(active.run))[0]['result'] == 'promoted'
        promoted = active.pin('recall-test', 'owner').closure.id
        from colonymind.turns.erasure_hooks import notify_source_erasure
        await asyncio.to_thread(notify_source_erasure, ['owner-recall'])
        assert driver.queries.get('owner-recall') is None
        assert not driver.evidence('owner-recall')
        assert active.pin('recall-test', 'owner').closure.id != promoted
        assert not active.loaded['recall-test'].comparator.records()
    finally:
        if server:
            server.should_exit = True
        if task:
            await asyncio.wait_for(task, 5)
        if listener is not None:
            listener.close()
        if client:
            await client.aclose()
        if active:
            active.close()
        if app:
            await app.store.close()
