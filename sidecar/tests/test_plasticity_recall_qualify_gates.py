"""Portable query revisions enter only the two benchmark query paths."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from colonymind.plasticity.query_adapter import save_revision
from colonymind.vector.indexes import EmbeddingIdentity

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    name = 'qualification_' + relative.replace('/', '_').replace('.', '_')
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    before = sys.path[:]
    sys.path.insert(0, str((ROOT / relative).parent))
    try:
        spec.loader.exec_module(module)
    finally:
        # Direct-script imports can prepend their own paths. Restore the whole
        # search path so public_memory/datasets cannot shadow HuggingFace datasets.
        sys.path[:] = before
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['benchmarks/public_memory/runner.py', 'benchmarks/source_recall/run.py'])
async def test_revision_query_path_identity_and_nonidentity(tmp_path, harness):
    from dataclasses import asdict
    before = sys.path[:]
    module = load(harness)
    assert sys.path == before
    class Pipeline:
        index_identity = EmbeddingIdentity('fixture', 'fixture', '1', 3)
        async def embed_query(self, text):
            return [.6, .8, 0.]
        async def embed_batch(self, texts):
            return [[.6, .8, 0.] for text in texts]
    pipeline = Pipeline()
    identity = asdict(pipeline.index_identity.document_identity)
    for changed in (False, True):
        u = np.zeros((3, 1), dtype=np.float32)
        v = np.ones((3, 1), dtype=np.float32) * .1
        if changed:
            u[0, 0] = .1
        path = tmp_path / str(changed)
        metadata = save_revision(path, u, v, bound=.05, document_identity=identity)
        adapted = module.QueryAdapterPipeline(pipeline, path)
        query = await adapted.embed_query('question')
        assert (query != await pipeline.embed_query('question')) == changed
        assert await adapted.embed_batch(['document']) == await pipeline.embed_batch(['document'])
        assert adapted.adapter_digest == metadata['digest']
        assert adapted.index_identity == pipeline.index_identity


def test_source_flag_parses_revision_directory(tmp_path):
    module = load('benchmarks/source_recall/run.py')
    args = module.parser().parse_args(['--state-dir', str(tmp_path), '--output', 'result.json',
                                      '--threshold', '.8', '--query-adapter', str(tmp_path / 'revision')])
    assert args.query_adapter == tmp_path / 'revision'


def test_public_flag_reaches_read(monkeypatch, tmp_path):
    module = load('benchmarks/public_memory/runner.py')
    observed = []
    monkeypatch.setattr(module, '_read', lambda args: observed.append(args.query_adapter) or 0)
    assert module.main(['read', '--dataset', 'beam', '--stores', 'stores', '--capture', 'capture',
                        '--work', 'work', '--query-adapter', str(tmp_path)]) == 0
    assert observed == [tmp_path]


@pytest.mark.asyncio
async def test_source_harness_identity_preserves_fixture_selection(tmp_path, monkeypatch):
    from dataclasses import asdict
    from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider
    from colonymind.vector.reranker import OpenAIAPIRerankerProvider, RerankResult
    from colonymind.plasticity.recall_qualify import environment
    module = load('benchmarks/source_recall/run.py')
    class Embedder(OpenAIAPIEmbeddingProvider):
        async def _embed(self, texts, **kwargs):
            self._served_model = 'fixture-embed'
            return [[1., 0., 0., 0.] for _ in texts]
    class Reranker(OpenAIAPIRerankerProvider):
        async def rerank(self, query, documents, top_k=10):
            return [RerankResult(i, .99, text) for i, text in enumerate(documents)]
    monkeypatch.setattr(module, 'OpenAIAPIEmbeddingProvider', Embedder)
    monkeypatch.setattr(module, 'OpenAIAPIRerankerProvider', Reranker)
    fixture = {'records': [{'id': 'source', 'at': '2026-09-01', 'scope': 'private',
                           'content': 'The blue crate is on shelf seven.'}],
               'queries': [{'id': 'query', 'query': 'Where is the blue crate?', 'principal': 'owner',
                            'as_of': '2026-09-05', 'expected': ['source'], 'tags': ['packing'], 'split': 'holdout'}]}
    source = tmp_path / 'fixture.json'
    source.write_text(json.dumps(fixture))
    config = {'COLONYMIND_EMBED_MODEL': 'fixture-embed', 'COLONYMIND_EMBED_DIMS': '4',
              'COLONYMIND_EMBED_BASE_URL': 'http://fixture', 'COLONYMIND_RERANKER_MODEL': 'fixture-rerank',
              'COLONYMIND_RERANKER_BASE_URL': 'http://fixture'}
    results = []
    for flag in ([], ['--query-adapter', str(tmp_path / 'identity')]):
        output = tmp_path / ('adapted.json' if flag else 'base.json')
        args = module.parser().parse_args(['--source-only', '--fixture', str(source), '--state-dir',
            str(tmp_path / 'state'), '--output', str(output), '--threshold', '.95', *flag])
        # The harness owns its standalone environment; restore it between in-process test runs.
        import os
        before = dict(os.environ)
        try:
            with environment(config):
                await module.run(config, args)
        finally:
            for key in set(os.environ) - set(before):
                os.environ.pop(key)
            os.environ.update(before)
        result = json.loads(output.read_text())
        results.append(result)
        if not flag:
            identity = EmbeddingIdentity(**result['model_identity'])
            save_revision(tmp_path / 'identity', np.zeros((4, 1), dtype=np.float32),
                          np.ones((4, 1), dtype=np.float32), bound=.05,
                          document_identity=asdict(identity.document_identity))
    base, adapted = results
    assert 'query_adapter' not in base['model_identity']
    assert adapted['model_identity']['query_adapter'] == 'identity'
    for key in ('context', 'assessment', 'semantic'):
        assert base['results'][0][key] == adapted['results'][0][key]
    assert base['results'][0]['assessment']['expected_found']


@pytest.mark.asyncio
async def test_public_harness_identity_preserves_context_and_score(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager, closing, ExitStack
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from dataclasses import asdict
    import gc
    import sqlite3
    import httpx
    from colonymind import vector
    from recall_route import recall_app, OWNER
    module = load('benchmarks/public_memory/runner.py')
    # Production context route, canonical sources, index and selector behind the runner's read call.
    app = await recall_app(monkeypatch, tmp_path / 'source', initiatives=0)
    monkeypatch.setattr('colonymind.util.temporal.now_utc',
                        lambda: datetime(2026, 9, 10, tzinfo=timezone.utc))
    save_revision(tmp_path / 'identity', np.zeros((3, 1), dtype=np.float32),
                  np.ones((3, 1), dtype=np.float32), bound=.05,
                  document_identity=asdict(app.embedder.index_identity.document_identity))
    class Clock:
        def set_clock(self, *args, **kwargs):
            pass
    monkeypatch.setattr(module, 'paired_body', Clock())
    monkeypatch.setattr(module.replay, 'load_manifest', lambda path: {'store': 'import_only'})
    monkeypatch.setattr(module, 'store_formation', lambda manifest: {'complete': True})
    monkeypatch.setattr(module, '_store_home', lambda *args: tmp_path / 'source')
    copytree = module.shutil.copytree
    def copy_fixture(source, *args, **kwargs):
        # Unlike a closed replay store, this fixture keeps serving its source.
        # Pin each SQLite ledger while copying so collection of idle catalog
        # connections cannot remove a WAL already listed by copytree.
        with ExitStack() as pins:
            for path in Path(source).glob('*.db'):
                connection = pins.enter_context(closing(sqlite3.connect(path)))
                connection.execute('SELECT count(*) FROM sqlite_master').fetchone()
            gc.collect()
            return copytree(source, *args, **kwargs)
    monkeypatch.setattr(module.shutil, 'copytree', copy_fixture)
    monkeypatch.setattr(module, '_store_end', lambda manifest: 0)
    monkeypatch.setattr(module, 'arm_config', lambda *args: None)
    monkeypatch.setattr(module.replay, 'open_instance', lambda *args: SimpleNamespace(owner=OWNER))
    monkeypatch.setattr(module.replay, 'gold', lambda *args: [{'turn:answer'}])
    @asynccontextmanager
    async def serve(*args, **kwargs):
        from onekey import KEY
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://test',
                                     headers={'Authorization': 'Bearer ' + KEY}) as client:
            yield SimpleNamespace(client=client, reranker=None)
    monkeypatch.setattr(module.replay, 'serve', serve)
    question = SimpleNamespace(id='question', conversation='conversation', type='fixture', abstention=False,
        asked_at=datetime(2026, 9, 10, tzinfo=timezone.utc), question='hydrofoil departure', evidence=[])
    try:
        base = await module.read_question(tmp_path, question, arm='import-only', budget=4000,
                                          work=tmp_path / 'base')
        adapted = await module.read_question(tmp_path, question, arm='import-only', budget=4000,
                                             work=tmp_path / 'adapted', query_adapter=tmp_path / 'identity')
        assert base['sections'] == adapted['sections']
        base_score, adapted_score = module.score(base), module.score(adapted)
        assert adapted_score.pop('query_adapter') == 'identity'
        assert base_score == adapted_score
        assert vector.get_pipeline() is app.embedder
    finally:
        await app.store.close()
