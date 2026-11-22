"""Canonical collection and selection shared by recall and explicit search."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Awaitable, Callable, Sequence

from .recall import MEMORY_PACKET_LIMIT, source_candidates, pack_memory_context
from .trace import NULL_TRACE
from . import producers, recall_exit

logger = logging.getLogger(__name__)

# Recall's own character budget when the environment names none (``COLONYMIND_RECALL_CONTEXT_MAX_CHARS``). The
# packet rides in the user turn and the host replays it as history on every later turn. An explicit search keeps
# DEFAULT_RECALL_CHARS. A turn's recall takes what the context budget has past REST_CHARS, the room the other
# sections keep, and never less than DEFAULT_RECALL_CHARS (``turn_recall_chars``).
DEFAULT_RECALL_CHARS = 4000
REST_CHARS = 6000


def turn_recall_chars(inline_chars: int | None = None) -> int | None:
    """The character budget of a turn's recall: the context budget (``context.budget_chars``, or the host's inline
    allowance when that is smaller) less ``REST_CHARS``, never less than ``DEFAULT_RECALL_CHARS``. None when the
    environment names the budget."""
    if os.environ.get('COLONYMIND_RECALL_CONTEXT_MAX_CHARS', '').strip():
        return None
    from . import blocks
    ceiling = int(blocks.settings()["budget_chars"])
    allowance = ceiling if inline_chars is None else min(int(inline_chars), ceiling)
    return max(DEFAULT_RECALL_CHARS, allowance - REST_CHARS)


#: The reciprocal-rank fusion constant: a hit at rank r in a lane adds 1 / (RRF_K + r).
RRF_K = 60


@dataclass
class CollectedSources:
    """What the lanes found. ``hits`` is every lane fused; ``lexical_hits`` the keyword lanes fused, the evidence
    recall has without a semantic index; ``semantic`` the vector lane's state (``ready``, ``unavailable``,
    ``failed``); ``lanes`` each lane's own hits, best first (a timed lane's once selection has run it)."""
    ledger: Any
    contact_id: str
    session_id: str
    watermark: int | None
    hits: list[dict[str, Any]] = field(default_factory=list)
    media: list[dict[str, Any]] = field(default_factory=list)
    semantic: str = 'unavailable'
    lexical_hits: list[dict[str, Any]] = field(default_factory=list)
    lanes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    curated_recall: Any = field(default=None, repr=False, compare=False)
    curated_protection: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class LaneQuery:
    """What every lane searches with: the question, the scope, the ledger and the optional semantic index; for a
    timed lane also the question's time (``beliefs.source_time.MemoryTimeQuery``) and what the other lanes found."""
    ledger: Any
    query: str
    contact_id: str
    session_id: str
    vector_store: Any = None
    embedding_pipeline: Any = None
    time_query: Any = None
    found: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    recall_request: Any = None


@dataclass
class LaneResult:
    """A lane's hits, best first; for a lane that also finds images, the images; and its state."""
    hits: list[dict[str, Any]]
    media: list[dict[str, Any]] = field(default_factory=list)
    state: str = 'ready'


@dataclass(frozen=True)
class Lane:
    """One retrieval lane of ``collect_sources``: its name in a recall trace and how it searches.

    ``keyword``: it reads exact words from the ledger, with no model, so its hits are also the keyword evidence
    (``lexical_hits``). ``optional``: a failure leaves the other lanes' evidence (``state`` ``failed``); any other
    lane's failure fails the collection. ``media``: it also finds images by their descriptions, traced as
    ``<name>_media``. ``timed``: it searches the question's time, which is read once the contact's time zone is
    known (``select_memory``), and reads the other lanes' ranks: it runs there, after them, only when the question
    names a span, and is fused and traced there with them."""
    name: str
    search: Callable[[LaneQuery], Awaitable[LaneResult]]
    keyword: bool = False
    optional: bool = False
    media: bool = False
    timed: bool = False


async def _keyword_lane(query: LaneQuery) -> LaneResult:
    # The ledger opens its own connection; large lexical scans must not hold the HTTP event loop while transport
    # intake waits for its response.
    return LaneResult(await producers.read(query.ledger.search_sources, query.query, contact_id=query.contact_id,
                                              session_id=query.session_id, limit=10, default=[]))


async def _stemmed_lane(query: LaneQuery) -> LaneResult:
    return LaneResult(await producers.read(query.ledger.search_stems, query.query, contact_id=query.contact_id,
                                              session_id=query.session_id, limit=10, default=[]))


async def _claim_key_lane(query: LaneQuery) -> LaneResult:
    # The words of claims' keys (subject, predicate, value): a hit is the message words the claim quotes, which
    # recall expands into the claim's bundle (``turns.claim_keys``).
    return LaneResult(await producers.read(query.ledger.search_claim_keys, query.query,
                                              contact_id=query.contact_id, session_id=query.session_id, limit=10, default=[]))


async def _vector_lane(query: LaneQuery) -> LaneResult:
    from colonymind.turns.source_vectors import SourceVectors
    sources, media = await SourceVectors(query.ledger, query.vector_store, query.embedding_pipeline).search(
        query.query, contact_id=query.contact_id, session_id=query.session_id, limit=15,
        **({'recall_request': query.recall_request} if query.recall_request is not None else {}))
    store = query.vector_store
    ready = (store is not None and query.embedding_pipeline is not None
             and getattr(store, 'catalog', None) is not None)
    return LaneResult(sources, media, 'ready' if ready else 'unavailable')


async def _window_lane(query: LaneQuery) -> LaneResult:
    from . import time_lane
    return LaneResult(await producers.read(
        time_lane.window_hits, query.ledger, query=query.query, contact_id=query.contact_id,
        session_id=query.session_id, time_query=query.time_query, vector_hits=query.found.get('vector', ()), default=[]))


#: The lanes, in the order they are fused (a tie in fused rank keeps this order) and traced. A lane added anywhere else
#: is invisible to a recall trace.
LANES: tuple[Lane, ...] = (
    Lane('lexical', _keyword_lane, keyword=True),
    Lane('vector', _vector_lane, optional=True, media=True),
    Lane('stemmed', _stemmed_lane, keyword=True),
    Lane('claim_key', _claim_key_lane, keyword=True),
    # The question's span of time (``memory.time_lane``): its sources, whatever their words.
    Lane('window', _window_lane, optional=True, timed=True),
)


def fuse(lanes: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion of the lanes' hits, keeping exact source lineage: a hit is one excerpt of one message
    (source, message hash, speaker, text), and one found by several lanes carries every lane's fields."""
    scores: dict[tuple, float] = {}
    rows: dict[tuple, dict[str, Any]] = {}
    for hits in lanes:
        for rank, row in enumerate(hits, 1):
            key = (row['turn_id'], row.get('source_message_hash'), row['role'], row['content'])
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank)
            rows[key] = {**rows.get(key, {}), **row}
    return [rows[key] for key in sorted(scores, key=scores.get, reverse=True)]


@dataclass
class MemoryPacket:
    selected: list[dict[str, Any]]
    content: str
    source_refs: list[dict[str, str]]
    watermark: int | None
    retrieval: dict[str, str]
    annotation_checks: list[dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return {'content': self.content, 'count': len(self.selected),
                'source_refs': self.source_refs, 'watermark': self.watermark,
                'retrieval': self.retrieval, 'annotation_checks': self.annotation_checks}


async def collect_sources(ledger, *, query: str, contact_id: str, session_id: str,
                          vector_store=None, embedding_pipeline=None, trace=NULL_TRACE,
                          recall_request=None) -> CollectedSources:
    """Read exact scoped sources; optional semantic failure keeps lexical evidence.

    Every lane of ``LANES`` searches at once, and the lanes are fused here; each lane's ranks go to ``trace``
    (``memory.trace``) where it is fused: a lane fused anywhere else would be invisible to a recall trace. A timed
    lane runs later, in ``select_memory``, where the question's time is read, and is fused and traced there.
    """
    from colonymind.memory.live_recall import LIVE
    from . import curated
    with curated.strict_read(ledger), LIVE.running():  # background indexing leaves the model endpoint to a live recall
        return await _collect_sources(ledger, query=query, contact_id=contact_id, session_id=session_id,
                                      vector_store=vector_store, embedding_pipeline=embedding_pipeline, trace=trace,
                                      recall_request=recall_request)


async def _collect_sources(ledger, *, query, contact_id, session_id, vector_store, embedding_pipeline, trace,
                           recall_request=None):
    from . import curated
    healthy = curated.recall_protection(ledger)
    collected = CollectedSources(ledger, contact_id, session_id,
                                await producers.read(curated.watermark, ledger, contact_id) if ledger is not None else None,
                                curated_protection=healthy)
    if not healthy() or ledger is None or not query.strip():
        return collected
    request = LaneQuery(ledger, query, contact_id, session_id, vector_store, embedding_pipeline,
                        recall_request=recall_request)
    lanes = [lane for lane in LANES if not lane.timed]
    # The lanes run at once: the query is embedded while the keyword searches run, not after them. Each lane is
    # read as it ends, not in registry order, so a failed lane that is not optional fails the collection at once
    # even while a lane registered before it still runs. That failure, or a cancelled collection, cancels the
    # other lanes and waits for them to end.
    running = [asyncio.ensure_future(lane.search(request)) for lane in lanes]
    results: dict[str, LaneResult] = {}
    try:
        pending = set(running)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for lane, task in zip(lanes, running):
                if task not in done:
                    continue
                try:
                    results[lane.name] = task.result()
                except Exception as exc:
                    recall_exit.failed(exc, "search")
                    if not lane.optional:
                        raise
                    if lane.name == 'vector' and recall_request is not None:
                        recall_request.reset()
                    results[lane.name] = LaneResult([], [], 'failed')
                    logger.debug('Recall lane %s unavailable (%s); retaining the other lanes', lane.name,
                                 type(exc).__name__)
    except BaseException:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    for lane in lanes:
        result = results[lane.name]
        if result.state == 'failed':
            continue
        trace.lane(lane.name, result.hits)
        if lane.media:
            trace.lane(f'{lane.name}_media', result.media, media=True)
    collected.lanes = {lane.name: results[lane.name].hits for lane in lanes}
    collected.hits = fuse(list(collected.lanes.values()))
    collected.lexical_hits = fuse([results[lane.name].hits for lane in lanes if lane.keyword])
    collected.media = [row for lane in lanes if lane.media for row in results[lane.name].media]
    collected.semantic = results['vector'].state
    return collected


async def _timed_lanes(collected: CollectedSources, *, query: str, time_query, trace) -> None:
    """Run the timed lanes for a question that names a span, after the others, and fuse every lane again in
    ``LANES`` order (``collected.hits``); each timed lane's ranks go to ``trace`` here, where it is fused."""
    timed = [lane for lane in LANES if lane.timed]
    if collected.ledger is None or not query.strip() or not timed or time_query.window() is None:
        return
    request = LaneQuery(collected.ledger, query, collected.contact_id, collected.session_id,
                        time_query=time_query, found=dict(collected.lanes))
    added = False
    for lane in timed:
        try:
            result = await lane.search(request)
        except Exception as exc:
            recall_exit.failed(exc, "search")
            if not lane.optional:
                raise
            logger.debug('Recall lane %s unavailable (%s); retaining the other lanes', lane.name, type(exc).__name__)
            continue
        trace.lane(lane.name, result.hits)
        collected.lanes[lane.name] = result.hits
        added = True
    if added:
        # A collection built by hand (no lanes of its own) counts its hits as one lane.
        base = [] if any(lane.name in collected.lanes for lane in LANES if not lane.timed) else [collected.hits]
        collected.hits = fuse(base + [collected.lanes[lane.name] for lane in LANES if lane.name in collected.lanes])


async def select_memory(collected: CollectedSources, *, query: str, selector,
                        timezone_name=None, current_work_available=False, limit=MEMORY_PACKET_LIMIT,
                        now=None, trace=NULL_TRACE, time_query=None, max_chars=None) -> MemoryPacket:
    """Apply the existing projections, corrections, ranking and shared budget once.

    The caller supplies an authenticated audience. All source candidates come
    from the canonical ledger. ``max_chars`` is the packet's character budget
    (None: the environment's, else ``DEFAULT_RECALL_CHARS``).
    ``now`` pins the time query; the recall benchmark replays fixtures at
    their recorded date. ``time_query`` is the question's time when the
    caller read it already (an explicit search's own window or ``as_of``);
    otherwise it is read here from the words. When it names a span, the timed
    lanes run first (the date-window lane). A ``trace``
    (``memory.trace.RecallTrace``) is handed to the selector, which then
    accepts one (``RecallSelector`` does).
    """
    async def read():
        healthy = getattr(collected, 'curated_protection', None)
        if healthy is not None and not healthy():
            return MemoryPacket([], '', [], collected.watermark, {}, [])
        # A collected cohort belongs to its captured withdrawal cursor. Do not
        # replay ranking against a later cursor or certify the old cohort anew.
        if (collected.ledger is not None and collected.watermark is not None
                and curated.watermark(collected.ledger, collected.contact_id) != collected.watermark):
            if trace:
                trace.recall(collected, limit=limit, max_chars=0)
                trace.packet([], '')
            return MemoryPacket([], '', [], collected.watermark, {}, [])
        from colonymind.memory.live_recall import LIVE
        from .curated_health import recalling
        with recalling(collected.curated_recall) as scope, LIVE.running():
            collected.curated_recall = scope
            return await _select_memory(collected, query=query, selector=selector, timezone_name=timezone_name,
                                        current_work_available=current_work_available, limit=limit, now=now,
                                        trace=trace, time_query=time_query, max_chars=max_chars)
    from colonymind.memory import curated
    return await curated.strict_withdrawal_fenced_read(collected.ledger, read, empty=MemoryPacket([], "", [], collected.watermark, {}, []))


def _prepare_memory(collected, *, query, time_query, current_work_available, now, trace):
    from colonymind.beliefs.source_time import filter_unstructured
    ledger = collected.ledger
    scope = {"contact_id": collected.contact_id, "session_id": collected.session_id}
    beliefs = []
    quotations = source_candidates(collected.hits)
    if ledger is not None:
        from colonymind.beliefs.source_projection import SourceClaimProjection
        from colonymind.memory.selection import current_work_query
        from colonymind.turns.media import SourceMedia
        projection = SourceClaimProjection(ledger)
        options = {'classify_work_replies': current_work_available and current_work_query(query),
                   'include_conversation_inputs': True}
        beliefs, quotations = projection.prepare_context(beliefs, collected.hits, **scope, time_query=time_query,
                                                         **options)
        if time_query.supplied:
            # A caller's window never hides what the lanes found, a message read through its claim included.
            from .time_lane import outside_window
            quotations.extend(outside_window(projection, collected.hits, quotations, **scope,
                                             time_query=time_query, now=now, **options))
        media_store = SourceMedia(ledger)
        media_hits = media_store.search(query, **scope)
        trace.lane('lexical_media', media_hits, media=True)
        quotations.extend(filter_unstructured(
            media_store.named_locators(query, collected.hits, **scope), time_query))
        media_by_id = {row['id']: row for row in media_hits + collected.media}
        quotations.extend(filter_unstructured(list(media_by_id.values()), time_query))
    else:
        beliefs = filter_unstructured(beliefs, time_query)
    if ledger is not None:
        from colonymind.turns.source_annotations import expand, current_candidates
        beliefs = expand(ledger, beliefs, **scope)
        quotations = expand(ledger, quotations, covered=beliefs, **scope)
        # Lexical hits were read before this point (while the query was embedded).
        # A source or message removed since has no current annotation membership; do
        # not emit its old text with empty references after expansion skips it.
        quotations = [row for row in quotations if not row.get('source_turn_id')
                      or row['source_turn_id'] in {
                          ref['source_id'] for ref in row.get('_annotation_source_refs', [])}]
        quotations = [row for row in quotations if row.get('kind') != 'media_locator'
                      or {key: row['source_read'][key] for key in ('source_id', 'source_version')}
                      in row.get('_annotation_source_refs', [])]
        # Preserve the independent lexical order through claim/correction
        # expansion. Only exact, canonically checked message membership can
        # carry it; another message in the same source does not inherit rank.
        lexical_ranks = {}
        for rank, hit in enumerate(collected.lexical_hits):
            lexical_ranks.setdefault((hit['turn_id'], hit.get('source_message_hash')), rank)
        for row in beliefs + quotations:
            ranks = [lexical_ranks[(source_id, message_hash)]
                     for source_id, hashes in row.get('_annotation_message_hashes', {}).items()
                     for message_hash in hashes if (source_id, message_hash) in lexical_ranks]
            if ranks:
                row['_lexical_rank'] = min(ranks)
    if ledger is not None:
        from .curated import strict_candidates as candidates
        beliefs, quotations = candidates(ledger, collected.contact_id, query, beliefs, quotations)
    return beliefs, quotations


async def _select_memory(collected, *, query, selector, timezone_name, current_work_available, limit,
                         now, trace, time_query, max_chars=None) -> MemoryPacket:
    from colonymind.beliefs.source_time import interpret_time_query, filter_unstructured
    from colonymind.util import temporal
    ledger = collected.ledger
    scope = {'contact_id': collected.contact_id, 'session_id': collected.session_id}
    beliefs = []
    now = now or temporal.now_utc()
    if time_query is None:
        time_query = interpret_time_query(query, now=now, timezone_name=timezone_name)
    await _timed_lanes(collected, query=query, time_query=time_query, trace=trace)
    if max_chars is None:
        try:
            max_chars = int(os.environ.get('COLONYMIND_RECALL_CONTEXT_MAX_CHARS', str(DEFAULT_RECALL_CHARS)))
        except (TypeError, ValueError):
            max_chars = DEFAULT_RECALL_CHARS
    max_chars = max(0, min(max_chars, 24000))
    trace.recall(collected, limit=limit, max_chars=max_chars)
    beliefs, quotations = await producers.read(_prepare_memory, collected, query=query,
        time_query=time_query, current_work_available=current_work_available, now=now, trace=trace,
        default=([], []))
    selected, content = await selector.select_context(query, beliefs, quotations, limit=limit,
        current_work_available=current_work_available, max_chars=max_chars,
        **({'trace': trace} if trace else {}))
    if ledger is not None:
        from colonymind.turns.source_annotations import current_candidates
        retained = await producers.read(current_candidates, ledger, selected, **scope, default=[])
        from .curated import strict_current_candidates as curated_current
        retained = await producers.read(curated_current, ledger, retained,
                                        contact_id=collected.contact_id, default=[])
        if len(retained) != len(selected):
            trace.rechecked(selected, retained)
            selected, content = pack_memory_context(retained, limit=limit, max_chars=max_chars, trace=trace)
    trace.packet(selected, content)
    source_ids = []
    for memory in selected:
        source_ids.extend(memory.get('source_turn_ids') or [])
        if memory.get('source_turn_id'):
            source_ids.append(memory['source_turn_id'])
        uri = str(memory.get('source_uri') or '')
        if uri.startswith('turn:'):
            source_ids.append(uri[5:])
    refs = await producers.read(ledger.source_references, source_ids, **scope) if ledger is not None else []
    refs_by_id = {ref['source_id']: ref for ref in refs}
    for memory in selected:
        # Keep the checked exact correction revision, never rebind old text to
        # a surviving version after an erasure.
        for ref in memory.get('_annotation_source_refs', []):
            refs_by_id[ref['source_id']] = ref
    checks = [{'source_refs': row['_annotation_source_refs'],
               'message_hashes': row['_annotation_message_hashes'],
               'annotation_ids': sorted(row['_annotation_ids'])}
              for row in selected if row.get('_annotation_source_refs')]
    return MemoryPacket(selected, content, list(refs_by_id.values()), collected.watermark,
                        {'semantic': collected.semantic}, checks)
