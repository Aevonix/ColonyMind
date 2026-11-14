"""Finite neutral comparison, using actual current ColonyMind retrieval paths.

Canonical SQLite/Lance and model calls are real, and the arms run the production
path (``collect_sources`` then ``select_memory``) over a disposable state directory.
"""
import argparse
import asyncio
import base64
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

# Also works from a checkout without installing the package itself.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'sidecar'))
from assessment import assess
from colonymind.router.router import LLMRouter
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.source_vectors import SourceVectors
from colonymind.turns.media import SourceMedia
from colonymind.vector.config import EmbeddingConfig
from colonymind.vector.embedder import EmbeddingPipeline
from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider
from colonymind.vector.reranker import OpenAIAPIRerankerProvider
from colonymind.vector.indexes import IndexCatalog
from colonymind.vector.store import VectorStore
from colonymind.memory.search import collect_sources, select_memory
from colonymind.memory.curated import forget_sources
from colonymind.qualification.recall_capture import SelectionCapture
from colonymind.plasticity.recall_qualify import QueryAdapterPipeline
from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
from colonymind.beliefs.source_projection import PRIORS_BY_MEANING_FLAG, SourceClaimProjection

# The packet measured is the one a host's turn receives (``/v1/host/context/assemble`` recalls with no count limit):
# behind a verified cutoff whose rerank returned, every candidate that passed it, in rank order within the budget;
# five when the cutoff is unverified or the rerank missed (``RecallSelector.select_context``). An explicit search
# names its own limit; a turn does not, so neither does this harness.
TURN_PACKET = {'kind': 'turn', 'limit': None, 'max_chars': 6000}


def image_message():
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (320, 160), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 35, 100, 120), fill='red')
    draw.ellipse((205, 40, 280, 115), fill='blue')
    output = io.BytesIO(); image.save(output, format='PNG')
    return {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Please retain this reference image.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()}},
    ]}


def environment(*, source_only=False):
    # Only explicit benchmark variables are read; no deployment config loader.
    required = ('EMBED_BASE_URL', 'EMBED_MODEL', 'EMBED_DIMS',
                'RERANKER_BASE_URL', 'RERANKER_MODEL')
    if not source_only:
        required += ('CHAT_BASE_URL', 'CHAT_MODEL')
    missing = [name for name in required if not os.environ.get('COLONYMIND_BENCH_' + name)]
    if missing:
        raise ValueError('Missing benchmark variables: ' + ', '.join('COLONYMIND_BENCH_' + name for name in missing))
    names = (*required, 'EMBED_API_KEY', 'RERANKER_API_KEY', 'CHAT_API_KEY',
             'RERANKER_PROMPT_STYLE', 'EMBED_QUERY_INSTRUCTION', 'CHAT_WEIGHT_REVISION',
             'RERANKER_REVISION', 'RECALL_INDEX_GENERATION')
    return {'COLONYMIND_' + name: os.environ['COLONYMIND_BENCH_' + name]
            for name in names if 'COLONYMIND_BENCH_' + name in os.environ}


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def prepare_sources(ledger, fixture):
    """Import exact fixture roles and real annotations without oracle fields."""
    for record in fixture['records']:
        role = record.get('role', 'user')
        if role not in {'user', 'assistant', 'tool'}:
            raise ValueError('Source-only fixtures support user, assistant or tool text')
        ledger.record_source(record['id'], contact_id='owner', session_id='neutral-corpus',
            messages=[{'role': role, 'content': record['content']}],
            occurred_at=record['at']+'T12:00:00+00:00', derive_claims=False)
    for note in fixture.get('annotations', []):
        ref, = ledger.source_references([note['target']], contact_id='owner', session_id='later')
        ledger.append_source_annotation(contact_id='owner', session_id='neutral-corpus',
            annotation_id=note['id'], source_id=note['target'], source_version=ref['source_version'],
            excerpt=note['excerpt'], correction=note['correction'], author_principal=note['author_principal'])


def priors_environment(on):
    """The projection's own switch of the priors' vector lane, as an install exports it
    (``mind.faculties.priors_by_meaning``): ``--no-priors-by-meaning`` turns it off, and nothing else changes."""
    return {PRIORS_BY_MEANING_FLAG: '1' if on else '0'}


def claim_projection(ledger, projections):
    """The claim projection extraction runs through, with the source vector search an install's claim worker has:
    its prior records come from the keyword lane and, while the projection's switch is on (``priors_environment``),
    from the vector lane too."""
    return SourceClaimProjection(ledger, vectors=projections)


async def extract(ledger, claims, projections, router, records, *, progress=None):
    """Record each fixture source and extract its claims, in fixture order. The sources recorded so far are
    embedded first, as an install's vector lane has embedded earlier messages by the time a later one is extracted,
    so the vector lane of its priors can find them when its switch is on."""
    for index, record in enumerate(records):
        ledger.record_source(record['id'], contact_id='owner', session_id='neutral-corpus',
            messages=[{'role': 'user', 'content': record['content']}], occurred_at=record['at'] + 'T12:00:00+00:00')
        while await projections.process_one():
            pass
        await claims.process_one(router)
        if progress is not None:
            progress(index)


def lexical_only(collected):
    """The same collection with semantic recall off: what production selects without an embedder."""
    return replace(collected, hits=list(collected.lexical_hits), media=[], semantic='unavailable')


async def run(config, args):
    fixture_path = args.fixture
    fixture = json.loads(fixture_path.read_text())
    if fixture.get('distractors', {}).get('count', 0):
        raise ValueError('This bounded harness requires explicit records, not generated distractors')
    if len(fixture['records']) > 120 or len(fixture['queries']) > 96:
        raise ValueError('At most 120 sources and 96 queries per run')
    tmp = args.state_dir.resolve()
    manifest_path = tmp / 'benchmark-state.json'
    resumed = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    # Never repurpose an existing deployment or an unmarked database directory.
    if resumed is None and tmp.exists() and any(tmp.iterdir()):
        raise ValueError('Use a new empty disposable state directory')
    tmp.mkdir(parents=True, exist_ok=True)
    identity = {'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                'extraction_model': config.get('COLONYMIND_CHAT_MODEL'),
                'extraction_weight_revision': config.get('COLONYMIND_CHAT_WEIGHT_REVISION', 'unknown'),
                'query_instruction': config.get('COLONYMIND_EMBED_QUERY_INSTRUCTION',
                    'Instruct: Given a search query, retrieve relevant memories that answer it\nQuery: ')}
    if args.source_only:
        identity['source_only'] = True
    else:
        # Which lanes offered the extractor its prior records: a store extracted with one is never resumed with
        # the other.
        identity['priors_by_meaning'] = args.priors_by_meaning
    if resumed and resumed.get('identity') != identity:
        raise ValueError('Fixture or extraction declaration changed; use a new state directory')
    if resumed is None:
        resumed = {'identity': identity, 'prepared': False, 'captures': []}
        save(manifest_path, resumed)
    if not args.source_only:
        os.environ.update(priors_environment(args.priors_by_meaning))
    os.environ.update(COLONYMIND_STATE_DIR=str(tmp), COLONYMIND_RECALL_HYBRID='on', COLONYMIND_RECALL_RERANK='on',
        COLONYMIND_RECALL_RERANK_TIMEOUT_MS='1200', COLONYMIND_RECALL_RERANK_MIN_SCORE=str(args.threshold),
        COLONYMIND_RECALL_CONTEXT_MAX_CHARS=str(TURN_PACKET['max_chars']),
        COLONYMIND_EMBED_QUERY_INSTRUCTION=identity['query_instruction'])
    router = LLMRouter() if not args.source_only else None
    host = {'provider': 'local', 'models': {}, 'modelPool': {'bench': {
        'model': config.get('COLONYMIND_CHAT_MODEL'), 'baseUrl': config.get('COLONYMIND_CHAT_BASE_URL'),
        'apiKey': config.get('COLONYMIND_CHAT_API_KEY', ''),
        'weightRevision': config.get('COLONYMIND_CHAT_WEIGHT_REVISION', 'unknown'),
        'maxTokens': 1400}}, 'functionRoles': {'extraction': ['bench']}}
    if os.environ.get('COLONYMIND_BENCH_LOCAL_HOSTS'):
        host['localHosts'] = os.environ['COLONYMIND_BENCH_LOCAL_HOSTS'].split(',')
    if router is not None:
        router.configure(host)
    model = config.get('COLONYMIND_CHAT_MODEL')
    provider=OpenAIAPIEmbeddingProvider(EmbeddingConfig(provider='openai_api', model_id=config['COLONYMIND_EMBED_MODEL'], dimensions=int(config['COLONYMIND_EMBED_DIMS'])))
    provider.configure(config['COLONYMIND_EMBED_BASE_URL'],config.get('COLONYMIND_EMBED_API_KEY',''))
    pipeline=EmbeddingPipeline(provider); await pipeline.warmup()
    query_pipeline = (QueryAdapterPipeline(pipeline, args.query_adapter)
                      if getattr(args, 'query_adapter', None) is not None else pipeline)
    embedding_identity = asdict(pipeline.index_identity)
    if resumed.get('embedding_identity') not in (None, embedding_identity):
        raise ValueError('Embedding identity changed; use a new state directory')
    resumed['embedding_identity'] = embedding_identity
    save(manifest_path, resumed)
    reranker=OpenAIAPIRerankerProvider(config['COLONYMIND_RERANKER_MODEL'])
    reranker.configure(config['COLONYMIND_RERANKER_BASE_URL'], config.get('COLONYMIND_RERANKER_API_KEY',''), config.get('COLONYMIND_RERANKER_PROMPT_STYLE',''))
    calibration={**reranker.calibration_metadata(), 'weights_revision':'unverified', 'embedding_identity':pipeline.index_identity.fingerprint,
                 'candidate_format':args.ranking_format}
    if args.source_only:
        # Same metadata fields and correction representation as serving recall.
        # The explicit trial cutoff is not newly qualified by this stamp.
        os.environ.update({key: config[key] for key in ('COLONYMIND_EMBED_MODEL', 'COLONYMIND_EMBED_DIMS')})
        for key in ('COLONYMIND_RERANKER_REVISION', 'COLONYMIND_RECALL_INDEX_GENERATION'):
            os.environ[key] = config.get(key, 'unverified')
        calibration = provider_calibration_metadata(reranker)
    os.environ['COLONYMIND_RECALL_RERANK_CALIBRATION']=calibration_fingerprint(calibration)
    calls=[]
    async def rerank(query, documents, top_k):
        result=await reranker.rerank(query,documents,top_k=top_k)
        return [asdict(row) for row in result]
    selector=SelectionCapture(rerank,calibration,calls,ranking_format=args.ranking_format)
    ledger=TurnIdempotencyLedger(tmp/'turn-idempotency.db')
    store=VectorStore(str(tmp/'lancedb'),identity=pipeline.index_identity,catalog=IndexCatalog(ledger))
    await store.connect(pipeline.dimensions); await store.ensure_collections(pipeline.dimensions)
    projections=SourceVectors(ledger,store,pipeline)
    claims=claim_projection(ledger,projections)
    captures=resumed['captures']; original_complete=router.complete if router is not None else None
    async def captured(**kwargs):
        start=time.perf_counter(); response=await original_complete(**kwargs)
        captures.append({'input':kwargs['messages'][-1]['content'],'output':response.content,
            'model':response.model_id,'ms':(time.perf_counter()-start)*1000})
        return response
    if router is not None:
        router.complete=captured
    # Corpus event times are input evidence. No expected labels, supersession,
    # confidence, or contradiction flags enter extraction or retrieval.
    if args.source_only and not resumed['prepared']:
        prepare_sources(ledger, fixture)
    def progress(index):
        if (index+1)%12==0: print(f"Actual extraction {index+1}/{len(fixture['records'])}",flush=True)
    await extract(ledger, claims, projections, router,
                  [] if resumed['prepared'] or args.source_only else fixture['records'], progress=progress)
    resumed.update(prepared=True, captures=captures)
    save(manifest_path, resumed)
    # Explicit fixture forget requests use the shared curated source fence.
    # Unlinked derived-summary fixture rows remain independent, intentionally.
    deleted=[row['id'] for row in fixture['records'] if row.get('deleted')]
    if deleted: forget_sources(ledger, 'owner', source_ids=deleted)
    # The source projections are the one semantic index; every arm reads them.
    while await projections.process_one(): pass
    pipeline = query_pipeline
    projections.pipeline = query_pipeline
    arms = ('canonical_hybrid',) if args.source_only else ('lexical_only','canonical_hybrid')
    results=[]
    queries = [q for q in fixture['queries'] if args.split is None or q['split'] == args.split]
    for index,q in enumerate(queries):
        contact=q['principal']; as_of=datetime.fromisoformat(q['as_of']+'T18:00:00+00:00')
        # One shared query embedding; report its measured cost separately.
        start=time.perf_counter(); await pipeline.embed_query(q['query']); query_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        collected=await collect_sources(ledger,query=q['query'],contact_id=contact,session_id='later',
                                        vector_store=store,embedding_pipeline=pipeline)
        collection_ms=(time.perf_counter()-start)*1000
        for arm in arms:
            start=time.perf_counter()
            candidates=lexical_only(collected) if arm=='lexical_only' else collected
            packet=await select_memory(candidates,query=q['query'],selector=selector,timezone_name='UTC',
                                       limit=TURN_PACKET['limit'],now=as_of)
            results.append({'query_id':q['id'],'split':q['split'],'tags':q['tags'],'arm':arm,
                'assessment':assess(q,packet.selected,fixture['records']),'context':packet.content,
                'replay':selector.last_replay,'semantic':collected.semantic if arm!='lexical_only' else 'unavailable',
                'selection_ms':(time.perf_counter()-start)*1000,'query_embedding_ms':query_ms,
                'collection_ms':collection_ms})
        if (index+1)%12==0: print(f"Actual retrieval {index+1}/{len(fixture['queries'])}",flush=True)
    summaries={}
    for arm in arms:
        summaries[arm]={}
        for split in ('development','holdout'):
            rows=[row for row in results if row['arm']==arm and row['split']==split]
            if not rows: continue
            assessments=[row['assessment'] for row in rows]
            summaries[arm][split]={'cases':len(rows),'strict_pass':sum(a['strict_pass'] for a in assessments),
                'expected_found':sum(a['expected_found'] for a in assessments),
                'mean_expected_recall':statistics.mean(a['recall'] for a in assessments if a['recall'] is not None)
                    if any(a['recall'] is not None for a in assessments) else None,
                'abstention_cases':sum(a['abstained'] is not None for a in assessments),
                'abstention_pass':sum(a['abstained'] is True for a in assessments),
                'forbidden_hits':sum(bool(a['forbidden']) for a in assessments),
                'conflicts_marked':sum(a['conflict_marked'] is True for a in assessments),
                'selection_p50_ms':statistics.median(row['selection_ms'] for row in rows),
                'missing_expected':[row['query_id'] for row in rows if not row['assessment']['expected_found']]}
            labeled = [a for a in assessments if 'useful_packet_pass' in a]
            if labeled:
                summaries[arm][split]['usefulness'] = {
                    'labeled_cases':len(labeled), 'useful_packet_pass':sum(a['useful_packet_pass'] for a in labeled),
                    'source_utility_pass':sum(a['source_utility_pass'] for a in labeled),
                    'irrelevant_selected':sum(a['relevance']['irrelevant_selected'] for a in labeled),
                    'unlabeled_selected':sum(len(a['relevance']['unlabeled']) for a in labeled)}
    media_results=[]
    if not args.source_only:
        # Fresh caption paraphrases, separate from the frozen corpus and its scores.
        ledger.record_source('fresh-image',contact_id='image-owner',session_id='image-session',messages=[image_message()])
        media_store=SourceMedia(ledger); job=media_store.claim_job()
        # This description is the output of the already-qualified image loop. This
        # benchmark measures caption retrieval, not new visual recognition quality.
        if job is not None:
            media_store.finish(job,description='A red rectangle is on the left and a blue circle on the right, on white.',model='previously-qualified-neutral-image-description')
        while await projections.process_one(): pass
        for query in ('Azure disc beside crimson quadrilateral', 'Circular object adjacent scarlet polygon', 'What was the pictured shape on the right?'):
            lexical=media_store.search(query,contact_id='image-owner',session_id='different')
            _,semantic=await projections.search(query,contact_id='image-owner',session_id='different')
            for arm, candidates in [('lexical',lexical),('hybrid',list({row['id']:row for row in lexical+semantic}.values()))]:
                selected,context,replay=await selector.select(query,[],candidates,limit=TURN_PACKET['limit'],
                                                              max_chars=TURN_PACKET['max_chars'])
                media_results.append({'query':query,'arm':arm,'candidates':len(candidates),'returned':bool(selected),'context':context,'replay':replay})
    with ledger._connect() as conn:
        job_status={row[0]:row[1] for row in conn.execute('SELECT status,count(*) FROM source_claim_jobs GROUP BY status')}
        claim_count=conn.execute('SELECT count(*) FROM source_claims').fetchone()[0]
    output={'summary':summaries,'results':results,'caption_results':media_results,'model_identity':asdict(pipeline.index_identity),
        'source_claim_job_status':job_status,'source_claim_count':claim_count,
        'extraction_model':model,'calibration':{k:v for k,v in calibration.items() if k != 'endpoint'},
        'fixed_threshold':args.threshold,'state_dir':str(tmp),'ranking_format':calibration['candidate_format'],
        'packet':dict(TURN_PACKET),
        'source_only':args.source_only,'split':args.split,
        'priors_by_meaning':None if args.source_only else args.priors_by_meaning,
        'fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),'calls':calls,
        'selection_sources': {str(path.relative_to(ROOT.parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(),
                         ROOT.parents[1] / 'sidecar/colonymind/memory/search.py',
                         ROOT.parents[1] / 'sidecar/colonymind/memory/selection.py',
                         ROOT.parents[1] / 'sidecar/colonymind/memory/recall.py')},
        'limits':['Default corpus: 120 frozen neutral sources, 96 queries, 24 holdout. A supplied smaller fixture is a smoke test.',
            'Actual local extraction/embeddings/reranker and canonical SQLite/Lance through the production collect_sources/select_memory path; lexical_only is that path with semantic recall unavailable.',
            'The packet is a host turn\'s: no count limit behind the verified cutoff, five when it is unverified or the rerank missed; 6000 characters, and no host allowance beside it.',
            'Each source is extracted after the sources before it are embedded, so its prior records come from the keyword and vector lanes as on an install (--no-priors-by-meaning: the keyword lane alone).',
            'Public/team fixture annotations do not invent shared authority: sources belong to fixture owner; six guest privacy queries expect abstention.',
            'Synthetic query_generation labels cannot substitute for a real embedding swap. Equal-dimension incompatibility is covered by real-Lance controlled tests separately.',
            'Unlinked historical derived-summary fixture records remain independent sources, not retroactively invented lineage.',
            'Threshold is an explicit operator input, not a universal score; immutable remote weights remain unknown unless separately declared.',
            'One shared query embedding measured separately; arm timings include actual selection but cached embedding. No production-scale latency or ANN benchmark.',
            'Fresh caption paraphrases use the prior neutral qualified description; they are text-caption search, not image embeddings or new visual understanding.']}
    if getattr(args, 'query_adapter', None) is not None:
        output['model_identity']['query_adapter'] = pipeline.adapter_digest
    if args.source_only:
        output['limits'] = [
            'Source-only canonical SQLite/Lance retrieval: lexical10 plus semantic15, actual selector candidate20, the turn packet (no count limit behind the verified cutoff, else five) within 6000 characters.',
            'Exact fixture tool/user/assistant text is imported; this does not test native observation nomination or ordinary conversation formation.',
            'Owner annotations use the actual source ledger. All corpus records belong to one synthetic owner; authority and concurrent mutations are not exercised.',
            'No extraction, media, contact-fact, native request assembly or generated final answer is exercised.',
            'The explicit trial cutoff and matching configuration stamp are not a new calibration qualification; returned weights remain unverified unless independently attested.',
            'Independent labels measure selected evidence usefulness and junk separately from eligibility; complete evidence does not guarantee a truthful answer.',
            'Use a new disposable state for a new embedding identity or fixture. Held-out labels must not select thresholds or source/query representations.']
    path=args.output; save(path,output)
    await provider.close()
    print(json.dumps({'summary':summaries,'captions':[{k:r[k] for k in ('query','arm','returned')} for r in media_results],'artifact':str(path)},indent=2),flush=True)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixtures.json')
    parser.add_argument('--source-only', action='store_true', help='Canonical source retrieval only: exact fixture roles and annotations, no extraction or captions')
    parser.add_argument('--split', choices=('development', 'holdout'), help='Run only one frozen query partition')
    parser.add_argument('--state-dir', type=Path, required=True, help='New disposable directory, or its marked extraction state to reuse')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-adapter', type=Path, help='Portable global query revision directory')
    parser.add_argument('--threshold', type=float, required=True, help='Run-specific declared reranker cutoff, never selected from holdout answers')
    parser.add_argument('--ranking-format', choices=('grounded-quotation-bundles-v1', 'verbose-claim-json'), default='grounded-quotation-bundles-v1')
    parser.add_argument('--no-priors-by-meaning', dest='priors_by_meaning', action='store_false',
                        help='The ablation of the priors\' vector lane: extraction offers prior records from the keyword lane alone')
    return parser


def main():
    parser_ = parser()
    args = parser_.parse_args()
    if not __import__('math').isfinite(args.threshold):
        parser_.error('threshold must be finite')
    asyncio.run(run(environment(source_only=args.source_only), args))


if __name__ == '__main__':
    main()
