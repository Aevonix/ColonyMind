"""Replaceable semantic projections of canonical quotations and image descriptions.

Jobs share the source ledger. Search is scoped before nearest-neighbor selection;
every result is re-read from its current canonical message or asset description.
Vector text is never an authoritative source, and retrieval never resolves truth.
"""
from __future__ import annotations

from colonymind.memory import producers, recall_exit, turn_budget

import asyncio
from contextlib import closing, contextmanager
from contextvars import ContextVar
import hashlib
import json
import time
import uuid

FORMAT = 'canonical-source-chunks-v1'
CHUNK = 2000
STRIDE = 1800


def initialize(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS source_vector_jobs (
        turn_id TEXT PRIMARY KEY,generation_id TEXT NOT NULL DEFAULT '',
        cursor INTEGER NOT NULL DEFAULT 0,status TEXT NOT NULL DEFAULT 'pending',
        lease_token TEXT NOT NULL DEFAULT '',lease_until REAL NOT NULL DEFAULT 0,
        next_attempt REAL NOT NULL DEFAULT 0,error TEXT)''')


def enqueue(conn, turn_id):
    conn.execute('''INSERT INTO source_vector_jobs(turn_id) VALUES (?)
        ON CONFLICT(turn_id) DO UPDATE SET generation_id='',cursor=0,status='pending',
        lease_token='',lease_until=0,next_attempt=0,error=NULL''', (turn_id,))


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


# The turn's query vectors (``turn_query_vectors``): one embedding per query text and pipeline for the turn, shared by
# the recall's vector lane and the lesson match (``P/mind/lessons.py``), whichever asks first.
_TURN_QUERY_VECTORS: ContextVar[dict | None] = ContextVar('colonymind_turn_query_vectors', default=None)


@contextmanager
def turn_query_vectors():
    """Keep this turn's query vectors for the block (and the tasks it starts), keyed by the pipeline and the hash of
    the query text; cleared when the block ends, an embedding still running then cancelled. Nothing outlives the
    turn."""
    memo: dict = {}
    token = _TURN_QUERY_VECTORS.set(memo)
    try:
        yield memo
    finally:
        _TURN_QUERY_VECTORS.reset(token)
        for entry in memo.values():
            if not entry['future'].done():
                entry['abandoned'] = True
                entry['future'].cancel()
        memo.clear()


def _retrieved(future):
    # A failed embedding nobody waited for is still retrieved, so it is not reported as never retrieved.
    if not future.cancelled():
        future.exception()


async def query_vector(pipeline, query, timeout):
    """The query's vector (``pipeline.embed_query``) within ``timeout`` seconds. Inside ``turn_query_vectors`` the
    turn embeds a text once: a second caller waits on the same call, which is one attempt, as a live query gets. When
    the last caller waiting on it gives up, the call is abandoned (as a lone caller's timeout abandons it), and a
    later caller in the turn gets a ``TimeoutError`` at once rather than a second attempt. The call is marked
    abandoned before it is cancelled, since a cancelled task is done only after the loop runs it again: a caller
    that comes in between is refused too, rather than waiting on a call about to end cancelled. A caller the
    abandoned call ends under gets a ``TimeoutError``; only a caller that is itself cancelled sees a cancellation."""
    memo = _TURN_QUERY_VECTORS.get()
    if memo is None:
        return await asyncio.wait_for(pipeline.embed_query(query), timeout)
    key = (id(pipeline), _hash(str(query)))
    entry = memo.get(key)
    if entry is None:
        future = asyncio.ensure_future(pipeline.embed_query(query))
        future.add_done_callback(_retrieved)
        entry = memo[key] = {'future': future, 'waiting': 0, 'abandoned': False}
    future = entry['future']
    if entry['abandoned'] or future.cancelled():
        raise TimeoutError("this turn's embedding of the query was abandoned")
    entry['waiting'] += 1
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout)
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if (entry['abandoned'] or future.cancelled()) and not (task is not None and task.cancelling()):
            raise TimeoutError("this turn's embedding of the query was abandoned") from None
        raise
    finally:
        entry['waiting'] -= 1
        if not entry['waiting'] and not future.done():
            entry['abandoned'] = True
            future.cancel()


def _scope_key(scope, contact_id, session_id):
    return _hash(json.dumps([scope, contact_id, session_id if scope == 'session' else ''], separators=(',', ':')))


def _text(message):
    from colonymind.turns.audio import source_text
    return source_text(message.get('content'))


def _metadata(source, message_hash, *, kind, text, **extra):
    meta = {'source_projection': FORMAT, 'source_uri': 'turn:' + source['turn_id'],
            'source_turn_id': source['turn_id'], 'source_message_hash': message_hash,
            'contact_id': source['contact_id'], 'session_id': source['session_id'],
            'scope_key': _scope_key(source['scope'], source['contact_id'], source['session_id']),
            'kind': kind, 'content_hash': _hash(text), **extra}
    meta['projection_id'] = 'source-vector:' + _hash(json.dumps(meta, sort_keys=True, separators=(',', ':')))
    return meta


def chunks(conn, source):
    from colonymind.turns.idempotency import source_message_hash
    for message in json.loads(source['messages_json']):
        message_hash = source_message_hash(source['session_id'], message)
        text = _text(message)
        for start in range(0, len(text), STRIDE):
            chunk = text[start:start + CHUNK]
            if chunk.strip():
                yield chunk, _metadata(source, message_hash, kind='source_quote', text=chunk, start=start)
        for asset in conn.execute('''SELECT DISTINCT m.asset_hash,m.description FROM source_media m
            JOIN source_media_links l ON l.asset_hash=m.asset_hash
            WHERE l.turn_id=? AND l.message_hash=? AND m.status='complete' AND m.description IS NOT NULL''',
                                  (source['turn_id'], message_hash)):
            yield asset['description'], _metadata(source, message_hash, kind='media_description',
                text=asset['description'], asset_hash=asset['asset_hash'])


def hydrate(ledger, meta, *, contact_id=None, session_id=None):
    """Validate exact lineage and return current source bytes, never vector text."""
    from colonymind.memory import curated
    if curated.is_forgotten(ledger, meta):
        return None
    return _hydrate_current(ledger, meta, contact_id=contact_id, session_id=session_id)


def _hydrate_current(ledger, meta, *, contact_id=None, session_id=None):
    """Read exact current membership after the caller has checked its fence."""
    if meta.get('source_projection') != FORMAT:
        return None
    from colonymind.turns.idempotency import source_message_hash
    with closing(ledger._read_connect()) as conn:
        from colonymind.turns.source_attribution import is_invalidated
        if is_invalidated(conn, meta.get('source_turn_id')):
            return None
        source = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (meta.get('source_turn_id'),)).fetchone()
        if source is None or meta.get('contact_id') != source['contact_id'] or meta.get('session_id') != source['session_id']:
            return None
        if meta.get('source_uri') != 'turn:' + source['turn_id']:
            return None
        if contact_id is not None and (source['contact_id'] != contact_id or
                source['scope'] != 'person' and source['session_id'] != session_id):
            return None
        if meta.get('scope_key') != _scope_key(source['scope'], source['contact_id'], source['session_id']):
            return None
        message = next((m for m in json.loads(source['messages_json'])
                        if source_message_hash(source['session_id'], m) == meta.get('source_message_hash')), None)
        if message is None:
            return None
        common = {'turn_id': source['turn_id'], 'role': message['role'], 'session_id': source['session_id'],
                  'scope': source['scope'], 'occurred_at': source['occurred_at'], 'ingested_at': source['ingested_at'],
                  'source_message_hash': meta['source_message_hash'], 'retrieval_method': 'semantic'}
        from colonymind.turns.audio import evidence_metadata
        common.update(evidence_metadata(message))
        if meta.get('kind') == 'source_quote':
            start = meta.get('start')
            if not isinstance(start, int) or start < 0 or start % STRIDE:
                return None
            message_text = _text(message)
            text = message_text[start:start + CHUNK]
            if not text.strip() or _hash(text) != meta.get('content_hash'):
                return None
            return {**common, 'content': text,
                    **({'excerpt_truncated': True} if text != message_text else {})}
        if meta.get('kind') == 'media_description':
            asset = conn.execute('''SELECT m.* FROM source_media m JOIN source_media_links l ON l.asset_hash=m.asset_hash
                WHERE l.turn_id=? AND l.message_hash=? AND m.asset_hash=? AND m.status='complete' LIMIT 1''',
                (source['turn_id'], meta['source_message_hash'], meta.get('asset_hash'))).fetchone()
            if asset is None or not asset['description'] or _hash(asset['description']) != meta.get('content_hash'):
                return None
            return {**common, 'id': 'media:' + asset['asset_hash'], 'kind': 'media_description',
                    'source_uri': 'turn:' + source['turn_id'], 'source_turn_id': source['turn_id'],
                    'asset_id': 'sha256:' + asset['asset_hash'], 'epistemic_state': 'derived_unverified',
                    'description_model': asset['model'], 'description_version': asset['description_version'],
                    'content': asset['description'], 'relevance': 1 / 61}
    return None


class SourceVectors:
    def __init__(self, ledger, store, pipeline):
        self.ledger, self.store, self.pipeline = ledger, store, pipeline

    def backfill(self):
        # Startup only. Source-only projection never replays actions or inference
        # that creates beliefs, affect, authority or commitments.
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO source_vector_jobs(turn_id) SELECT turn_id FROM turn_sources')

    def status(self, contact_id):
        catalog = getattr(self.store, 'catalog', None)
        active = catalog.active() if catalog is not None else None
        generation = active['id'] if active else ''
        compatible = bool(active and self.pipeline is not None
                          and self.store.identity == self.pipeline.index_identity
                          and active['fingerprint'] == self.store.identity.fingerprint)
        with closing(self.ledger._connect()) as conn:
            rows = conn.execute('''SELECT j.status,j.generation_id,j.error FROM turn_sources s
                LEFT JOIN source_vector_jobs j ON j.turn_id=s.turn_id WHERE s.contact_id=?''', (contact_id,)).fetchall()
        complete = sum(row['status'] == 'complete' and row['generation_id'] == generation for row in rows)
        return {'index_state': 'compatible' if compatible else 'unverified_or_unavailable',
                'active_generation': generation or None, 'source_turns': len(rows),
                'projected_turns': complete, 'pending_turns': len(rows) - complete,
                'failed_turns': sum(bool(row['error']) for row in rows)}

    def _claim(self, generation):
        from colonymind.memory import curated
        now = time.time()
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = next((candidate for candidate in conn.execute('''SELECT j.* FROM source_vector_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id
                WHERE (j.status<>'complete' OR j.generation_id<>?) AND j.lease_until<=? AND j.next_attempt<=?
                ORDER BY j.next_attempt,s.ingested_at,j.turn_id''', (generation['id'], now, now)) if not curated.is_forgotten(self.ledger, candidate['turn_id'], conn=conn)), None)
            if row is None:
                return None
            cursor = row['cursor'] if row['generation_id'] == generation['id'] else 0
            token = uuid.uuid4().hex
            conn.execute("UPDATE source_vector_jobs SET generation_id=?,cursor=?,status='running',lease_token=?,lease_until=? WHERE turn_id=?",
                         (generation['id'], cursor, token, now + 60, row['turn_id']))
            return dict(row, cursor=cursor, generation_id=generation['id'], lease_token=token,
                        projected=row['generation_id'])

    def _finish(self, job, *, cursor, complete=False, error=None):
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute('''UPDATE source_vector_jobs SET cursor=?,status=?,lease_until=0,next_attempt=?,error=?
                WHERE turn_id=? AND lease_token=?''', (cursor, 'complete' if complete else 'pending',
                time.time() + 30 if error else (0 if complete else time.time()),
                error, job['turn_id'], job['lease_token']))

    async def _table(self, generation):
        # Called with the store's write lock held.
        from colonymind.vector.collections import Collection
        table = await self.store._table(Collection.CONVERSATIONS, write=True, generation=generation)
        if 'scope_key' not in (await table.schema()).names:
            await table.add_columns({'scope_key': 'CAST(NULL AS STRING)'})
        return table

    async def process_one(self, batch_size=16):
        from colonymind.vector.collections import Collection
        from colonymind.vector.indexes import IncompatibleIndex
        if self.store is None or self.pipeline is None or getattr(self.store, 'catalog', None) is None:
            return False
        if self.store.identity != self.pipeline.index_identity:
            return False
        # A live recall gets the model endpoint: no batch starts while one runs (up to a bound; then a small one).
        from colonymind.memory.live_recall import batch_size_beside_recall
        batch_size = await batch_size_beside_recall(batch_size)
        try:
            generation = self.store.catalog.write_generation(self.store.identity)
        except IncompatibleIndex:
            return False  # Unknown legacy stays lexical until an explicit rebuild.
        job = self._claim(generation)
        if job is None:
            return False
        try:
            with closing(self.ledger._connect()) as conn:
                source = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (job['turn_id'],)).fetchone()
                if source is None:
                    return True
                import itertools
                batch = list(itertools.islice(chunks(conn, source), job['cursor'], job['cursor'] + min(16, max(1, batch_size)) + 1))
            more = len(batch) > min(16, max(1, batch_size))
            batch = batch[:min(16, max(1, batch_size))]
            if batch:
                # The usage rows' task (``colonymind.usage``): a source's first embedding is the memory lane's cost
                # of its capture; re-embedding one indexed before, for a new index generation, is maintenance.
                from colonymind import usage
                task = 'source_embedding' if job['projected'] in ('', generation['id']) else 'source_reindex'
                with usage.embedding_task(task):
                    vectors = await asyncio.wait_for(self.pipeline.embed_batch([text for text, _ in batch]), 40)
                if len(vectors) != len(batch):
                    raise ValueError('source embedding cardinality mismatch')
                for vector in vectors:
                    self.store._validate_vector(vector)
                erased = []
                # Open the table and commit under the store's write lock: a
                # table rewrite after erasure replaces the files that a handle
                # opened earlier still points at, and would drop this commit.
                async with self.store.write_lock:
                    table = await self._table(generation)
                    for (text, meta), vector in zip(batch, vectors):
                        if not self.store._eligible(Collection.CONVERSATIONS, meta['projection_id'], meta):
                            continue
                        now = time.time()
                        meta['embedding_fingerprint'] = self.store.identity.fingerprint
                        row = {'id': meta['projection_id'], 'text': text, 'vector': vector, 'metadata': json.dumps(meta),
                               'scope_key': meta['scope_key'], 'modality': 'text', 'image_hash': '', 'image_ref': '',
                               'thumbnail_ref': '', 'caption': '', 'created_at': now, 'updated_at': now}
                        await table.merge_insert('id').when_matched_update_all().when_not_matched_insert_all().execute([row])
                        if not self.store._eligible(Collection.CONVERSATIONS, meta['projection_id'], meta):
                            erased.append(meta['projection_id'])
                for projection_id in erased:
                    await self.store.delete(Collection.CONVERSATIONS, projection_id)
            self._finish(job, cursor=job['cursor'] + len(batch), complete=not more)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._finish(job, cursor=job['cursor'], error=type(exc).__name__)
        return True

    async def search(self, query, *, contact_id, session_id, limit=10, recall_request=None):
        with turn_budget.producer_writes():
            try:
                return await self._search(query, contact_id=contact_id, session_id=session_id,
                                          limit=limit, recall_request=recall_request)
            except BaseException as exc:
                recall_exit.failed(exc)
                raise

    async def _search(self, query, *, contact_id, session_id, limit, recall_request):
        from colonymind.vector.collections import Collection
        from colonymind.vector.indexes import IncompatibleIndex
        if not contact_id or self.store is None or self.pipeline is None or getattr(self.store, 'catalog', None) is None:
            return [], []
        if self.store.identity != self.pipeline.index_identity:
            raise IncompatibleIndex('The source query pipeline and index identity do not match')
        # Validate generation before spending inference, and never read an
        # unscoped top-K followed by filtering away another person's results.
        generation = await producers.read(self.store.catalog.read_generation, self.store.identity)
        table = await self.store._table(Collection.CONVERSATIONS, generation=generation)
        if 'scope_key' not in (await table.schema()).names:
            raise IncompatibleIndex('Source index has no participant scope projection; lexical recall remains available')
        keys = [_scope_key('person', contact_id, session_id), _scope_key('session', contact_id, session_id)]
        # One attempt, within the turn's budget (``memory.turn_budget``): a hung embedder costs the turn about
        # 1.5 s, and recall keeps its keyword evidence.
        from colonymind.memory import turn_budget
        timeout = turn_budget.embed_timeout()
        if timeout <= 0:
            recall_exit.code('deadline', 'embed')
            raise TimeoutError('no time left in the turn for the query embedding')
        try:
            vector = await query_vector(self.pipeline, query, timeout)
        except BaseException as exc:
            recall_exit.failed(exc, "embed")
            raise
        options = dict(limit=min(25, max(1, limit)),
            filter='scope_key IN (' + ','.join(self.store._quoted(key) for key in keys) + ')')
        if recall_request is None:
            hits = await self.store.search(Collection.CONVERSATIONS, vector, **options)
        else:
            try:
                hits = await self._owner_search(vector, recall_request, options, table,
                                                 contact_id=contact_id, session_id=session_id)
            except BaseException as exc:
                recall_exit.failed(exc, "owner_search")
                raise
        from colonymind.memory import curated
        return await producers.read(lambda: curated.fenced_read(self.ledger,
            lambda: self._hydrate_hits(hits, contact_id=contact_id, session_id=session_id),
            empty=([], [])), default=([], []))

    def _hydrate_hits(self, hits, *, contact_id, session_id):
        sources, media = [], []
        for hit in hits:
            current = hydrate(self.ledger, hit.metadata, contact_id=contact_id, session_id=session_id)
            if current is not None:
                (media if current.get('kind') == 'media_description' else sources).append(current)
        return sources, media

    async def _owner_search(self, vector, request, options, table, *, contact_id, session_id):
        """Pair local searches, then log through Runtime.serve under the checked joint pin.

        Async index reads never hold the runtime's thread lock. If a promotion or
        erasure moved the pointer while they ran, retry from the new joint closure.
        Three attempts are a builder assumption; contention leaves lexical recall.
        """
        from dataclasses import asdict
        import numpy as np
        from colonymind.plasticity.learners.query_adapter import transform_identity
        from colonymind.plasticity.query_adapter import weights_digest
        from colonymind.plasticity.revisions import Pinned
        from colonymind.vector.collections import Collection
        runtime, driver = request.runtime, request.driver
        binding = runtime.loaded[request.name]
        if asdict(self.store.identity.document_identity) != driver.document_identity:
            recall_exit.code('error', 'owner_search', 'identity', 'ValueError')
            raise ValueError('query adapter document identity does not match the index')
        memo_key = ('hits', _hash(json.dumps([vector, options], sort_keys=True)))

        def check_erasure():
            if producers.expired():
                raise TimeoutError('query adapter read deadline')
            if driver.pending_erasure():
                raise ValueError('pending source erasure prevents learned recall')
            if driver.erased(request.turn_id):
                raise ValueError('recall was erased')

        def pair():
            # Project summary's rollback without writing from a read worker.
            # record() applies/rechecks the comparator under the same lock.
            with runtime.lock:
                check_erasure()
                serving = runtime.pin(request.name, 'owner')
                previous = binding.controller.previous(serving.closure.id)
                pending = binding.store.revisions.pending_rollback()
                summary = binding.comparator._summary(serving.closure.id, previous.id)
                if pending or (serving.closure.mode != 'base' and summary['labelled']
                               and summary['lower_bound'] > binding.comparator.bar):
                    target = binding.store.revisions.get(pending['target']) if pending else previous
                    serving = Pinned(target, binding.store.revisions.state(target))
                    previous = binding.controller.previous(target.id)
                reference = Pinned(previous, binding.store.revisions.state(previous))
                request.pin(serving)
                return serving, reference

        def memoized():
            with runtime.lock:
                check_erasure()
                return request.memo[memo_key]

        if memo_key in request.memo:
            return await producers.read(memoized)

        def current_hits(hits):
            return [hit for hit in hits if hydrate(self.ledger, hit.metadata,
                    contact_id=contact_id, session_id=session_id) is not None]

        async def predict(pinned):
            state = pinned.state
            transform = transform_identity(state) if state else 'identity'
            key = ('vector', id(self.pipeline), memo_key, transform)
            if key not in request.memo:
                # The base arm preserves the provider's bytes, including its normalization.
                request.memo[key] = (driver.learner.apply(state, vector).tolist()
                                     if state and transform != 'identity' else vector)
            hits = await self.store.search(Collection.CONVERSATIONS, request.memo[key], **options)
            hits = await producers.read(current_hits, hits)
            if state and weights_digest(state['shown_U'], state['shown_V']) != 'identity' and hits:
                # Read existing passage vectors, never text or a new embedding.
                predicate = 'id IN (' + ','.join(self.store._quoted(hit.id) for hit in hits) + ')'
                rows = (await table.query().where(predicate).select(['id', 'vector']).to_arrow()).to_pylist()
                passages = {row['id']: row['vector'] for row in rows}
                if set(passages) != {hit.id for hit in hits}:
                    raise ValueError('shown passage vectors changed during recall')
                adversarial = await producers.read(lambda: [hit.id for hit in hits if (hydrate(self.ledger, hit.metadata,
                    contact_id=contact_id, session_id=session_id) or {}).get('role') != 'user'])
                scores = driver.learner.rank_scores(state, vector, passages,
                    shown=list(passages), adversarial=adversarial)
                hits.sort(key=lambda hit: scores[hit.id], reverse=True)
            return hits

        for _ in range(3):
            serving, reference = await producers.read(pair)
            served_hits = await predict(serving)
            reference_hits = await predict(reference)
            served_hits = await producers.read(current_hits, served_hits)
            reference_hits = await producers.read(current_hits, reference_hits)
            passages = {}
            for hit in served_hits + reference_hits:
                current = await producers.read(hydrate, self.ledger, hit.metadata,
                    contact_id=contact_id, session_id=session_id)
                if current is not None:
                    passages[hit.id] = {"id": hit.id, "source_id": current["turn_id"],
                        "message_hash": hit.metadata.get("source_message_hash"),
                        "source_kind": "owner" if current.get("role") == "user"
                        and current.get("scope") == "person" else "tool"}
            latest, latest_reference = await producers.read(pair)
            if (latest.closure.id != serving.closure.id
                    or latest_reference.closure.id != reference.closure.id):
                continue

            def record(served_hits=served_hits, reference_hits=reference_hits, passages=passages,
                       closure=serving.closure.id, previous=reference.closure.id):
                with runtime.lock:
                    if driver.pending_erasure() or driver.erased(request.turn_id):
                        recall_exit.captured('c')
                        return
                    binding.comparator.summary()
                    if (runtime.pin(request.name, 'owner').closure.id != closure
                            or binding.controller.previous(closure).id != previous):
                        recall_exit.captured('d')
                        return
                    paired = iter(([hit.id for hit in served_hits], [hit.id for hit in reference_hits]))
                    runtime.serve(request.name, request.turn_id, vector,
                        predict=lambda pinned, inputs: next(paired), requester='owner')
                    driver.capture(request.turn_id, np.asarray(vector, dtype=np.float32), list(passages.values()),
                        [hit.metadata['source_turn_id'] for hit in served_hits],
                        [hit.metadata['source_turn_id'] for hit in reference_hits])
                    recall_exit.captured('recorded')
            turn_budget.defer_write(record)
            recall_exit.queued()
            request.memo[memo_key] = served_hits
            return served_hits
        raise ValueError('query adapter changed during recall')
