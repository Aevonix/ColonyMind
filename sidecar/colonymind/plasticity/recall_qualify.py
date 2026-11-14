"""Produce local, reference-only recall qualification through the serving selector.

Input JSON freezes development, held_out, gate and canaries before inference.
Rows contain query_id, query, relevant source ids and no_answer. Canaries also
name forbidden source ids. Queries and retrieved text remain in memory only.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import time

from . import _io
from .learners.query_adapter import RecallRequest, transform_identity
from .revisions import Pinned


class QueryAdapterPipeline:
    """A portable global revision on queries only; document embeddings are unchanged."""
    def __init__(self, pipeline, directory):
        from .query_adapter import load_revision
        self.pipeline = pipeline
        self.revision = load_revision(directory, document_identity=asdict(pipeline.index_identity.document_identity))
        if self.revision.metadata.get('scope', 'global') != 'global':
            raise ValueError('query-adapter requires a global revision')
        self.adapter_digest = self.revision.metadata['digest']
        self.query_transform = ('identity' if self.adapter_digest == 'identity'
                                else 'query-adapter-1:' + self.adapter_digest)

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    async def embed_query(self, query):
        vector = await self.pipeline.embed_query(query)
        # Identity preserves the provider's exact bytes, as owner serving does.
        return vector if self.adapter_digest == 'identity' else self.revision.apply(vector).tolist()


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate(driver, payload, build):
    if payload.get('role') != 'gate_suite' or not payload.get('set_id'):
        raise ValueError('qualification requires a named gate_suite')
    rows = [row for split in ('development', 'held_out', 'gate', 'canaries') for row in payload[split]]
    ids = [row['query_id'] for row in rows]
    if any(not isinstance(ref, str) or not ref or any(char.isspace() for char in ref) for ref in ids):
        raise ValueError('query ids must be nonempty references without whitespace')
    if len(ids) != len(set(ids)):
        raise ValueError('all query ids must be distinct across splits')
    for row in rows:
        if (not isinstance(row.get('query'), str) or not row['query'].strip()
                or type(row.get('no_answer')) is not bool or not isinstance(row.get('relevant'), list)
                or any(not isinstance(ref, str) or not ref for ref in row['relevant'])
                or row['no_answer'] != (not row['relevant'])):
            raise ValueError('queries need text, relevant source ids and a consistent boolean no_answer')
    for split in ('development', 'held_out'):
        if not any(row['no_answer'] for row in payload[split]) or all(row['no_answer'] for row in payload[split]):
            raise ValueError('each calibration split needs answer and no-answer queries')
    if sum(row['no_answer'] for row in payload['held_out']) < 150:
        raise ValueError('held_out needs at least 150 distinct no-answer queries')
    if len(payload['gate']) < 1000:
        raise ValueError('gate needs at least 1000 distinct queries')
    if not payload['canaries'] or any(not row.get('forbidden') for row in payload['canaries']):
        raise ValueError('canaries must name forbidden source ids')
    dependencies = set(build.refs)
    for record in driver.records():
        if record['turn_id'] in build.refs:
            dependencies.update(record['dependencies'])
    if dependencies.intersection(ids):
        raise ValueError('qualification overlaps training references')
    # Read-only equivalent of registration, before any model calls or capture writes.
    for registered in driver.sets._sets():
        if registered['id'] == payload['set_id'] and (registered['role'] != 'gate_suite'
                or set(registered['refs']) != set(ids)):
            raise ValueError('set role or frozen items changed')
        if registered['role'] == 'sealed' and set(registered['refs']).intersection(ids):
            raise ValueError('qualification overlaps a sealed set')
    return ids


def _budget(payload, max_queries, max_seconds):
    if not math.isfinite(max_seconds) or max_seconds <= 0 or max_queries <= 0:
        raise ValueError('budgets must be positive and finite')
    count = len(payload['development']) + len(payload['held_out']) + 4 * (
        len(payload['gate']) + len(payload['canaries']))
    if count > max_queries:
        raise TimeoutError('qualification query budget exhausted')


async def qualify(driver, payload, *, scorer, max_queries=10000, max_seconds=3600):
    """Measure a pending build; publish one immutable capture only after all work succeeds.

    The limits count actual selections across arms, including canaries. They are
    operator budgets, not evidence thresholds. No promotion happens here.
    """
    from colonymind.memory.recall import calibration_fingerprint
    from colonymind.recall_cli import refit_query_adapter_cutoff
    _budget(payload, max_queries, max_seconds)
    start = time.monotonic()
    build, control = driver.build(), driver.control()
    _validate(driver, payload, build)
    if scorer.document_identity != driver.document_identity:
        raise ValueError('document identity does not match the learner')
    if driver.pending_erasure():
        raise ValueError('pending erasure prevents qualification')
    path = driver.store.directory / 'qualification' / (build.digest + '.json')
    if path.exists() or (driver.store.directory / 'cutoff-checks' / (build.digest + '.json')).exists():
        raise ValueError('revision already captured or consumed its held-out check')
    previous = driver.store.revisions.serving()
    if previous.state == build.digest:
        previous = driver.controller.previous(previous.id)
    reference_state = driver.store.revisions.state(previous)
    reference_cutoff = driver.cutoff(Pinned(previous, reference_state))
    transform = transform_identity(build.state)
    metadata = dict(scorer.metadata)
    fingerprint = calibration_fingerprint({**metadata, 'query_transform': transform})

    async def score(row, state, *, arm='candidate', cutoff=None):
        left = max_seconds - (time.monotonic() - start)
        if left <= 0:
            raise TimeoutError('qualification time budget exhausted')
        try:
            result = await asyncio.wait_for(scorer.score(row, state, cutoff=cutoff, arm=arm), left)
        except TimeoutError:
            raise TimeoutError('qualification time budget exhausted') from None
        scores, relevant = result['scores'], result['relevant']
        if (len(scores) != len(relevant) or any(type(value) not in (int, float)
                or not math.isfinite(value) for value in scores)
                or any(type(value) is not bool for value in relevant)
                or not isinstance(result.get('selected'), list)
                or any(not isinstance(value, str) for value in result['selected'])):
            raise ValueError('scorer returned invalid scores')
        return result

    captured = {}
    for split in ('development', 'held_out'):
        captured[split] = []
        for row in payload[split]:
            result = await score(row, build.state)
            captured[split].append({'query_id': row['query_id'], 'scores': result['scores'],
                'relevant': result['relevant'], 'no_answer': row['no_answer'],
                'calibration_fingerprint': fingerprint})
    cutoff = refit_query_adapter_cutoff(metadata, transform, captured['development'], captured['held_out'])
    gate = {'ids': [row['query_id'] for row in payload['gate'] + payload['canaries']]}
    for arm, state, threshold in (('base', driver.learner.base(), None),
            ('reference', reference_state, reference_cutoff), ('candidate', build.state, cutoff),
            ('control', control, cutoff)):
        gate[arm] = []
        for row in payload['gate']:
            result = await score(row, state, arm=arm, cutoff=threshold)
            selected = set(result['selected'])
            gate[arm].append(not selected if row['no_answer'] else bool(selected.intersection(row['relevant'])))
        exposures = {}
        for row in payload['canaries']:
            result = await score(row, state, arm=arm, cutoff=threshold)
            exposures[row['query_id']] = float(bool(set(result['selected']).intersection(row['forbidden'])))
            selected = set(result['selected'])
            gate[arm].append(not selected if row['no_answer'] else bool(selected.intersection(row['relevant'])))
        if arm in {'base', 'candidate'}:
            gate['canaries_' + arm] = exposures
    capture = {'state': build.digest, 'control_state': _io.state_digest(control), 'query_transform': transform,
        'document_identity': driver.document_identity, 'reference': previous.id, 'role': 'gate_suite',
        'set_id': payload['set_id'], 'metadata': metadata, **captured, 'gate': gate}
    # A concurrent rebuild, rollback or index/configuration change invalidates the work.
    if (driver.store.revisions.serving().id != previous.id or driver.build().digest != build.digest
            or _io.state_digest(driver.control()) != capture['control_state'] or driver.pending_erasure()):
        raise ValueError('learner changed during qualification')
    if dict(scorer.metadata) != metadata or scorer.document_identity != driver.document_identity:
        raise ValueError('scoring identity changed during qualification')
    if time.monotonic() - start >= max_seconds:
        raise TimeoutError('qualification time budget exhausted')
    with driver.controller.lock:
        if path.exists():
            raise ValueError('revision already captured its held-out check')
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.capture-')
        os.close(descriptor)
        try:
            _io.write_json(Path(temporary), capture)
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


class _ScoringStore:
    """Read the same index with global and shown factors, without live evidence writes."""
    def __init__(self, store, pipeline, driver, state, ledger, owner):
        self.store, self.pipeline, self.driver, self.state = store, pipeline, driver, state
        self.ledger, self.owner = ledger, owner

    def __getattr__(self, name):
        return getattr(self.store, name)

    async def search(self, collection, vector, **kwargs):
        from colonymind.plasticity.query_adapter import weights_digest
        from colonymind.turns.source_vectors import hydrate
        state = self.state
        transformed = (self.driver.learner.apply(state, vector).tolist()
                       if state and transform_identity(state) != 'identity' else vector)
        hits = await self.store.search(collection, transformed, **kwargs)
        # SourceVectors hydrates every returned hit before offering candidates.
        # Only shown-factor ranking needs a separate, earlier canonical read;
        # identity shown factors preserve the index's order unchanged.
        rank_shown = bool(state) and weights_digest(state['shown_U'], state['shown_V']) != 'identity'
        if rank_shown:
            hits = [hit for hit in hits if hydrate(self.ledger, hit.metadata,
                contact_id=self.owner, session_id='qualification') is not None]
        if rank_shown and hits:
            table = await self.store._table(collection)
            predicate = 'id IN (' + ','.join(self.store._quoted(hit.id) for hit in hits) + ')'
            rows = (await table.query().where(predicate).select(['id', 'vector']).to_arrow()).to_pylist()
            passages = {row['id']: row['vector'] for row in rows}
            if set(passages) != {hit.id for hit in hits}:
                raise ValueError('shown passage vectors changed during qualification')
            adversarial = [hit.id for hit in hits if (hydrate(self.ledger, hit.metadata,
                contact_id=self.owner, session_id='qualification') or {}).get('role') != 'user']
            scores = self.driver.learner.rank_scores(state, vector, passages, shown=list(passages),
                                                    adversarial=adversarial)
            hits.sort(key=lambda hit: scores[hit.id], reverse=True)
        return hits


class RecallScorer:
    """Instance collection, candidate formation, reranking and packet selection."""
    def __init__(self, driver, *, ledger, store, pipeline, reranker, owner):
        self.driver, self.ledger, self.store = driver, ledger, store
        self.pipeline, self.reranker, self.owner = pipeline, reranker, owner
        self.generation = store.catalog.read_generation(store.identity)['id']
        if store.identity != pipeline.index_identity:
            raise ValueError('embedding and index identity differ')

    @property
    def document_identity(self):
        return asdict(self.store.identity.document_identity)

    @property
    def metadata(self):
        from colonymind.memory.recall import provider_calibration_metadata
        if self.store.catalog.read_generation(self.store.identity)['id'] != self.generation:
            raise ValueError('index generation changed during qualification')
        return provider_calibration_metadata(self.reranker)

    async def score(self, row, state, *, cutoff=None, arm='candidate'):
        from colonymind.memory.recall import calibration_fingerprint
        from colonymind.memory.search import collect_sources, select_memory
        from colonymind.memory.selection import RecallSelector
        from colonymind.memory.trace import RecallTrace, _carried, _shown_id, _turns
        from colonymind.turns.source_vectors import turn_query_vectors
        request = RecallRequest(None, '', '')
        request.transform = transform_identity(state) if state else 'identity'
        metadata = {**self.metadata, 'query_transform': request.transform}
        # Raw calibration captures still run the full selector and force a scored call.
        if cutoff is not None:
            request.cutoff = {**cutoff, 'fingerprint': calibration_fingerprint(metadata)}
        async def rerank(query, documents, top_k):
            result = await self.reranker.rerank(query, documents, top_k=top_k)
            return [asdict(value) if hasattr(value, '__dataclass_fields__') else value for value in result]
        selector = RecallSelector(rerank, calibration_metadata=lambda: metadata)
        def sources(item):
            return set(_turns(item)) | {source for source, _ in _carried(item)}
        class QualificationTrace(RecallTrace):
            def __init__(self):
                super().__init__()
                self.references = {}

            def submitted(self, rows, *, bound):
                super().submitted(rows, bound=bound)
                self.references.update({_shown_id(self._see(item)): sources(item) for item in rows})
        trace = QualificationTrace()
        store = _ScoringStore(self.store, self.pipeline, self.driver, state, self.ledger, self.owner)
        values = {'COLONYMIND_RECALL_RERANK': 'on'}
        if cutoff is None and arm == 'candidate':
            values.update(COLONYMIND_RECALL_RERANK_MIN_SCORE='-1e30',
                          COLONYMIND_RECALL_RERANK_CALIBRATION=calibration_fingerprint(metadata))
        with environment(values), request.use(), turn_query_vectors():
            collected = await collect_sources(self.ledger, query=row['query'], contact_id=self.owner,
                session_id='qualification', vector_store=store, embedding_pipeline=self.pipeline, trace=trace)
            if collected.semantic != 'ready':
                raise ValueError('semantic recall unavailable during qualification')
            packet = await select_memory(collected, query=row['query'], selector=selector,
                                         timezone_name='UTC', trace=trace)
        report = trace.report()
        if report['rerank'] and report['rerank']['outcome'] in {'missed', 'off', 'shadow'}:
            raise ValueError('reranker did not score qualification')
        scored = [value for value in report['rows'] if value['submitted'] and value['rerank_score'] is not None]
        selected = {source for item in packet.selected for source in sources(item)}
        return {'scores': [value['rerank_score'] for value in scored],
                'relevant': [bool(trace.references.get(value['candidate'], set()).intersection(row['relevant']))
                             for value in scored],
                'selected': sorted(selected)}


async def run_command(args):
    """Open configured providers and the existing index, then capture the pending build."""
    from colonymind.config import apply_environment, load_config
    from colonymind import vector
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.vector import EmbeddingConfig, EmbeddingPipeline, make_provider
    from colonymind.vector.config import VectorStoreConfig
    from colonymind.vector.indexes import IndexCatalog
    from colonymind.vector.store import VectorStore
    from colonymind.vector.reranker import (OpenAIAPIRerankerProvider, CPURerankerProvider,
        CUDARerankerProvider, MLXRerankerProvider, NativeMLXRerankerProvider)
    from .nightly import Runtime
    config = load_config()
    payload = json.loads(Path(args.queries).read_text())
    _budget(payload, args.max_queries, args.max_seconds)
    derived = dict(os.environ)
    apply_environment(config, environ=derived)
    for option, variable in (('embed_url', 'COLONYMIND_EMBED_BASE_URL'),
                             ('rerank_url', 'COLONYMIND_RERANKER_BASE_URL')):
        if getattr(args, option, None):
            derived[variable] = getattr(args, option)
    if args.embed_url:
        derived['COLONYMIND_EMBED_PROVIDER'] = 'openai_api'
    with environment(derived):
        runtime = Runtime.from_config(config)
        provider = reranker = store = None
        previous_store, previous_pipeline = vector.get_store(), vector.get_pipeline()
        try:
            binding = runtime.loaded.get(args.learner)
            if binding is None or binding.declaration['implementation'] != 'query_adapter':
                raise ValueError('learner is not a loaded query_adapter')
            driver = binding.driver
            driver.attach(runtime)
            embedding = EmbeddingConfig.from_env()
            if embedding is None:
                raise ValueError('an embedder must be configured')
            provider = make_provider(embedding)
            if provider is None:
                raise ValueError('an embedder must be configured')
            if embedding.provider == 'openai_api':
                provider.configure(os.environ.get('COLONYMIND_EMBED_BASE_URL', ''),
                                   os.environ.get('COLONYMIND_EMBED_API_KEY', ''))
            pipeline = EmbeddingPipeline(provider)
            await pipeline.warmup()
            model = os.environ.get('COLONYMIND_RERANKER_MODEL')
            if not model:
                raise ValueError('a reranker must be configured')
            url = os.environ.get('COLONYMIND_RERANKER_BASE_URL')
            kind = os.environ.get('COLONYMIND_RERANKER_PROVIDER', '')
            if url or kind == 'openai_api':
                if not url:
                    raise ValueError('remote reranker requires an endpoint')
                reranker = OpenAIAPIRerankerProvider(model)
            else:
                if not kind:
                    from colonymind.vector.scanner import scan
                    kind = scan().gpu_type
                    if kind == 'mlx':
                        try:
                            __import__('mlx_lm')
                            kind = 'native_mlx'
                        except ImportError:
                            pass
                providers = {'cpu': CPURerankerProvider, 'cuda': CUDARerankerProvider,
                             'mlx': MLXRerankerProvider, 'native_mlx': NativeMLXRerankerProvider,
                             'none': CPURerankerProvider}
                if kind not in providers:
                    raise ValueError('unknown configured reranker provider')
                reranker = providers[kind](model)
            if url:
                reranker.configure(url, os.environ.get('COLONYMIND_RERANKER_API_KEY', ''),
                                   os.environ.get('COLONYMIND_RERANKER_PROMPT_STYLE', ''))
            await reranker.warmup()
            ledger = TurnIdempotencyLedger(config.home / 'turn-idempotency.db')
            directory = VectorStoreConfig.from_env().data_dir
            if not Path(directory).is_dir():
                raise ValueError('qualification requires an existing index')
            store = VectorStore(directory, identity=pipeline.index_identity,
                                catalog=IndexCatalog(ledger))
            await store.connect(pipeline.dimensions)
            scorer = RecallScorer(driver, ledger=ledger, store=store, pipeline=pipeline,
                                  reranker=reranker, owner=os.environ.get('COLONYMIND_OWNER_CONTACT_ID')
                                  or config.get('owner.contact_id'))
            # Driver training reads existing passage vectors through the instance's index.
            vector.set_store(store)
            vector.set_pipeline(pipeline)
            driver.period()
            return {'capture': str(await qualify(driver, payload, scorer=scorer,
                max_queries=args.max_queries, max_seconds=args.max_seconds))}
        finally:
            runtime.close()
            vector.set_store(previous_store)
            vector.set_pipeline(previous_pipeline)
            for resource in (store, provider, reranker):
                close = getattr(resource, 'close', None)
                if close is not None:
                    await close()
