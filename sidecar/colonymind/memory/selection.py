"""Shared bounded reranking for authorized belief and source candidates."""
from __future__ import annotations

from . import recall_exit

import asyncio
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

from .recall import RERANK_DOCUMENT_CHARS, RERANK_QUERY_CHARS  # noqa: F401 (the bounds recall sends the reranker)
from .recall import MEMORY_PACKET_LIMIT
from .trace import NULL_TRACE

# The most candidates a turn's recall sends the reranker in one call: it bounds the model's work as the retrieval
# lanes add candidates. It is independent of the context record cap, so raising that cap does not widen it. A caller that names its own limit (an explicit search) sends four times
# that limit, as before. Past the bound the candidates are chosen across their sessions (``spread_across_sessions``).
RERANK_SUBMISSION_LIMIT = 20

# Rerank scores this close are a tie, broken by the fused retrieval order (``tie_order``). A candidate that answers the
# question in the person's own words and the agent's restatement of them (or tool output repeating the same facts)
# can score within a hair of each other, and one candidate's score moves a little between two calls. The margin
# reorders near ties only: it changes no score, no candidate text and no cutoff decision, so it is not part of the
# calibrated configuration. What difference is a tie depends on the scale of one deployment's scores, which the
# verified configuration does not establish, so there is none by default: a deployment sets the margin it validated
# on its own queries with ``COLONYMIND_RECALL_RERANK_TIE_MARGIN``, and it applies only while a verified cutoff does.
RERANK_TIE_MARGIN = 0.0
_CALIBRATED = frozenset({"configuration_verified", "configuration_verified_weights_unverified"})


@dataclass(frozen=True)
class Offered:
    """What a packet's packer was offered: every candidate left after ranking (behind a verified cutoff, every one
    that passed it), in rank order, with the packet's count limit (None: none) and recall's own character budget."""
    rows: tuple
    limit: int | None
    max_chars: int


_OFFERED: ContextVar[list | None] = ContextVar("colonymind_recall_offered", default=None)


@contextmanager
def keep_offered() -> Iterator[list[Offered]]:
    """While open, a selection (``RecallSelector.select_context``) adds what its packer was offered to the list this
    yields. Context assembly opens it around a turn's recall: the host's inline allowance is known only once the
    context is assembled, and packing within it from what recall's own budget already kept cannot bring back a
    record that budget left out (a short answer behind a long procedure that now shrinks to a pointer)."""
    offered: list[Offered] = []
    token = _OFFERED.set(offered)
    try:
        yield offered
    finally:
        _OFFERED.reset(token)


_TURN_PACKETS: ContextVar[list | None] = ContextVar("colonymind_turn_packets", default=None)


@contextmanager
def turn_packets() -> Iterator[list]:
    """While open, the recall of a turn's context hands its packet over to the list this yields (``hand_over``)
    rather than finishing it itself: context assembly, which opens it, fits the packet within the host's allowance
    once every item kept whole beside it exists, and records what the packet then shows as served."""
    packets: list = []
    token = _TURN_PACKETS.set(packets)
    try:
        yield packets
    finally:
        _TURN_PACKETS.reset(token)


def hand_over(packet) -> bool:
    """Hand a turn's recalled packet to the context assembly that opened ``turn_packets``; False when none did."""
    packets = _TURN_PACKETS.get()
    if packets is None:
        return False
    packets.append(packet)
    return True


def _offer(rows, limit: int | None, max_chars: int) -> None:
    offered = _OFFERED.get()
    if offered is not None:
        offered.append(Offered(tuple(rows), limit, max_chars))


def rerank_tie_margin() -> float:
    """``COLONYMIND_RECALL_RERANK_TIE_MARGIN``, or the default (none) when it is unset or unreadable; never below 0."""
    raw = os.environ.get("COLONYMIND_RECALL_RERANK_TIE_MARGIN", "").strip()
    try:
        value = float(raw) if raw else RERANK_TIE_MARGIN
    except ValueError:
        value = RERANK_TIE_MARGIN
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else RERANK_TIE_MARGIN


def tie_order(rows: List[Dict[str, Any]], margin: float) -> List[Dict[str, Any]]:
    """Order ``rows`` (given in fused retrieval order) by relevance, treating a score within ``margin`` of the best
    remaining one as a tie: each place goes to the earliest fused candidate within ``margin`` of the best remaining
    score. No candidate goes ahead of one that scored more than ``margin`` higher; with no margin this is the
    relevance order, equal scores in fused order."""
    remaining, ordered = list(rows), []
    while remaining:
        best = max(row.get("relevance", 0) for row in remaining)
        index = next(i for i, row in enumerate(remaining) if row.get("relevance", 0) >= best - margin)
        ordered.append(remaining.pop(index))
    return ordered


def _source_of(row) -> str | None:
    """The source a candidate is anchored in: the one its words were read from, else the one its link names (an
    assertion bundle's ``source_uri``), else the first one it rests on."""
    if row.get("source_turn_id"):
        return str(row["source_turn_id"])
    uri = str(row.get("source_uri") or "")
    if uri.startswith("turn:"):
        return uri[5:]
    sources = row.get("source_turn_ids") or ()
    return str(sources[0]) if sources else None


def _session_of(row, sessions: Dict[str, str]) -> tuple[str, str]:
    """The conversation a candidate comes from, for the spread: its own session; else the session of the source it
    is anchored in, as the candidates read from that source say (``sessions``: the ledger's value, carried by each
    excerpt, pair and complete message); else that source; else the candidate itself. A projected assertion bundle
    carries no session of its own, and one conversation can state many properties."""
    if row.get("session_id"):
        return "session", str(row["session_id"])
    source = _source_of(row)
    if source is not None:
        return ("session", sessions[source]) if source in sessions else ("source", source)
    return ("id", str(row["id"])) if row.get("id") else ("row", str(id(row)))


def spread_across_sessions(rows: List[Dict[str, Any]], bound: int, *, last=None) -> List[Dict[str, Any]]:
    """The candidates the reranker is sent: at most ``bound`` of ``rows`` (given in fused order), in fused order.

    Up to ``bound`` rows, every one of them. Past it, they are chosen round robin across their sessions: each
    session in the order of its strongest candidate, one candidate per session each round, strongest first within
    it, so the overlapping chunks of one conversation cannot take the whole batch and leave every other one unread.
    A row ``last`` marks is chosen only after every other row. It only chooses: no row, score or cutoff changes, and
    what is not sent keeps its place for a selection without scores."""
    if len(rows) <= bound:
        return list(rows)
    late = [index for index, row in enumerate(rows) if last is not None and last(row)]
    sessions = {str(row["source_turn_id"]): str(row["session_id"]) for row in rows
                if row.get("source_turn_id") and row.get("session_id")}
    queues: Dict[tuple[str, str], List[int]] = {}
    for index, row in enumerate(rows):
        if last is None or not last(row):
            queues.setdefault(_session_of(row, sessions), []).append(index)
    chosen: List[int] = []
    depth = 0
    while len(chosen) < bound:
        turn = [queue[depth] for queue in queues.values() if depth < len(queue)]
        if not turn:
            break
        chosen.extend(turn[:bound - len(chosen)])
        depth += 1
    chosen.extend(late[:bound - len(chosen)])
    return [rows[index] for index in sorted(chosen)]


def current_work_query(query):
    """Recognize explicit present-work questions, retaining ambiguous requests.

    This is a narrow selection hint, not authority or a semantic classifier.
    History, comparisons and requests for instructions still need old answers.
    """
    if not isinstance(query, str) or len(query) > 8000:
        return False
    text = re.sub(r"^\s*\[[^\]\n]{1,512}\]\s*", "", query).strip().lower()
    if re.search(r"\b(yesterday|histor(?:y|ical)|previous(?:ly)?|earlier|before|then|"
                 r"last (?:time|week|month|year)|used to|compar(?:ed?|ing|ison)|versus|"
                 r"how (?:to|do|can|should)|procedure|playbook|recipe|steps|"
                 r"(?:give|provide|explain|show|need|include|use|follow|recall) (?:the |your |me )?instructions)\b", text):
        return False
    return bool(re.match(
        r"(?:what are you (?:building|developing)(?: for me)? (?:right now|now|currently)\b|"
        r"what are you (?:currently (?:doing|working on)|"
        r"(?:doing|working on) (?:right now|now|currently))\b|"
        r"what (?:work|tasks|sessions|jobs|crons|workers) (?:are|is) "
        r"(?:currently |now )?(?:running|active|in flight)\b)", text))


class RecallSelector:
    def __init__(self, rerank_fn=None, *, calibration_metadata=None, logger=None):
        self._rerank_fn = rerank_fn
        self._rerank_calibration_metadata = calibration_metadata
        self.logger = logger or logging.getLogger(__name__)

    async def select_context(self, query, beliefs, quotations, *, limit=MEMORY_PACKET_LIMIT, max_chars=6000,
                             current_work_available=False, trace=NULL_TRACE):
        """One rank fusion, reranking pass and budget after authority checks. ``trace`` (``memory.trace``) is told
        the fused order, what the reranker did and what the packer kept.

        ``limit`` is the most records an explicit search holds. None requests a context packet, bounded by
        ``context.recall_items`` and by ``max_chars``, whether or not a verified cutoff chose its records. A
        missed rerank keeps its smaller bound (``recall_stats.fallback_limit``)."""
        from .recall import fuse_candidates, pack_memory_context
        beliefs = [dict(row, kind="belief") for row in beliefs]
        # Semantic retrieval can return the same words from many turns. Collapse
        # only plain quotations by the same known speaker, after scoped claim
        # expansion and time filtering. Keep the first occurrence's intact
        # lineage; assertion/conflict bundles, dated events and uncertain
        # attribution survive. A time-qualified query may need both occurrences.
        unique, seen = [], set()
        omit_status_replies = current_work_available and current_work_query(query)
        for row in quotations:
            # Only canonical request/answer pairing can identify an old status
            # reply. Never use quotation text alone or a client owner flag.
            if (omit_status_replies
                    and row.get('_current_work_status_reply') is True
                    and row.get('kind') == 'source_quote' and row.get('role') == 'assistant'
                    and row.get('scope') == 'person'
                    and row.get('epistemic_state') == 'quotation'
                    and not any(row.get(key) for key in (
                        'atomic_evidence', 'validity_status', 'procedure_context', '_annotation_ids'))):
                continue
            if (row.get("kind") == "source_quote" and row.get("epistemic_state") == "quotation"
                    and not row.get("atomic_evidence") and row.get("scope") == "person"
                    and not row.get("validity_status")
                    and row.get("contact_id") and row.get("role") in {"user", "assistant"}):
                key = (row["contact_id"], row["role"], row["content"])
                if key in seen:
                    continue
                seen.add(key)
            unique.append(row)
        quotations = unique
        def repeats_request(row):
            # An exact repeat contributes no independent answer evidence. Keep
            # it available as supplementary context, without classifying other
            # questions, promoting speakers or changing its source bytes.
            return bool(isinstance(query, str) and query.strip()
                and row.get("kind") == "source_quote"
                and row.get("epistemic_state") == "quotation"
                and row.get("scope") == "person" and row.get("contact_id")
                and row.get("role") in {"user", "assistant"}
                and isinstance(row.get("content"), str)
                and row["content"].strip() == query.strip()
                and not any(row.get(key) for key in (
                    "atomic_evidence", "validity_status", "procedure_context",
                    "source_context", "_annotation_ids")))

        # Media is an independent candidate producer. Appending its first hit
        # after every text hit can exclude it from the bounded reranker before
        # relevance is assessed. Fuse its rank independently; it still shares
        # the same reranker, abstention threshold and final context budget.
        # Exact names locate admitted originals even when no passage contains
        # their visual answer. They share the final budget, not the calibrated
        # answer-passage cutoff. No generic lexical hit receives this treatment.
        locators = [row for row in quotations if row.get("kind") == "media_locator"]
        quotations = [row for row in quotations if row.get("kind") != "media_locator"]
        media = [row for row in quotations if row.get("kind") == "media_description"]
        text = [row for row in quotations if row.get("kind") != "media_description"]
        # Confidence in a belief and certainty that words were quoted are not
        # comparable truth scores. Preserve them as evidence metadata; select
        # across kinds by rank and semantic relevance alone.
        candidates = fuse_candidates(
            beliefs, text, limit=len(beliefs) + len(quotations),
            confidence_weighting=False, additional=(media,))
        # Do this before bounded reranking, so repeated requests do not crowd
        # out answer-bearing candidates before their relevance can be judged.
        candidates.sort(key=repeats_request)
        trace.fused(candidates)
        from . import blocks, turn_budget
        context_limit = int(blocks.settings()["recall_items"])
        packet_limit = min(MEMORY_PACKET_LIMIT, context_limit) if limit is None else (
            min(limit, context_limit) if turn_budget.current() is not None else limit)
        ranked, missed, _ = await self._reranked(
            query, candidates, packet_limit, confidence_weighting=False,
            candidate_limit=RERANK_SUBMISSION_LIMIT if limit is None else max(1, 4 * limit),
            last=repeats_request, trace=trace)
        # ``ranked`` is still in fused order. On a calibrated scale, near ties keep it (``tie_order``).
        if any(row.get('rerank_status') == 'scored' for row in ranked) and all(
                row.get('rerank_calibration') in _CALIBRATED for row in ranked):
            ranked = tie_order(ranked, rerank_tie_margin())
        else:
            ranked.sort(key=lambda row: row.get("relevance", 0), reverse=True)
        mode = os.environ.get('COLONYMIND_RECALL_RERANK', 'off').strip().lower()
        if mode != 'shadow' and not any(row.get('rerank_status') == 'scored' for row in ranked):
            # A source indexed by both producers wins reciprocal-rank fusion
            # over a freshly captured lexical-only fact. Without a working
            # reranker, alternate lexical and hybrid ranks without letting
            # duplicates spend a turn. Neither weak keyword matches nor index
            # coverage alone should monopolize the character budget. Shadow
            # mode keeps its original ordering for comparison.
            lexical = sorted((row for row in ranked if '_lexical_rank' in row),
                             key=lambda row: row['_lexical_rank'])
            orders = (iter(lexical), iter(ranked))
            fallback, seen = [], set()
            while True:
                before = len(fallback)
                for order in orders:
                    for row in order:
                        if row['id'] not in seen:
                            fallback.append(row)
                            seen.add(row['id'])
                            break
                if len(fallback) == before:
                    break
            ranked = fallback
        # A semantic reranker can score the identical question highest. Preserve
        # its abstention decisions and substantive order, then place only exact
        # unqualified request repeats after the remaining evidence.
        ranked.sort(key=repeats_request)
        from . import recall_stats
        if not missed or mode != 'on':
            # A context packet holds up to ``context.recall_items`` records within its character budget. A verified
            # cutoff changes relevance admission, never that cap. Exact media locators share the cap and the budget.
            fitted_limit = context_limit if limit is None else packet_limit
            _offer(locators + ranked, fitted_limit, max_chars)
            selected, content = pack_memory_context(locators + ranked, limit=fitted_limit,
                                                    max_chars=max_chars, trace=trace)
            return selected, content
        # A missed rerank skipped the calibrated cutoff: the fallback's fused order, up to ``MEMORY_PACKET_LIMIT`` in
        # a turn (an explicit search's own limit) until a deployment calibrates a smaller bound
        # (``recall_stats.fallback_limit``), and counted.
        _offer(locators + ranked, recall_stats.fallback_limit(packet_limit), max_chars)
        selected, content = pack_memory_context(locators + ranked, limit=recall_stats.fallback_limit(packet_limit),
                                                max_chars=max_chars, trace=trace)
        recall_stats.STATS.fallback(turn=turn_budget.current() is not None, rows=len(selected))
        return selected, content

    async def rerank(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        limit: int,
        *,
        strength_ranking: bool = False,
        confidence_weighting: bool = True,
        candidate_limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Cross-encoder rerank of filtered recall candidates, bounded and
        fail-open (``_reranked``, without its miss reason)."""
        ranked, _, _ = await self._reranked(
            query, memories, limit, strength_ranking=strength_ranking,
            confidence_weighting=confidence_weighting, candidate_limit=candidate_limit)
        return ranked

    async def _reranked(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        limit: int,
        *,
        strength_ranking: bool = False,
        confidence_weighting: bool = True,
        candidate_limit: int | None = None,
        last=None,
        trace=NULL_TRACE,
    ) -> tuple[List[Dict[str, Any]], str | None, bool]:
        """Cross-encoder rerank of filtered recall candidates, bounded and
        fail-open; why a rerank the configuration asks for did not score
        (``no_reranker``, ``budget``, ``busy``, ``timeout``, ``error`` or
        ``invalid``: ``recall_stats.MISS_REASONS``; None otherwise); and
        whether a verified calibrated cutoff decided which candidates are
        returned (the rerank scored, ``on``, and its configuration matched).
        Every such rerank is counted (``recall_stats``), and ``trace`` is told
        what it did.

        COLONYMIND_RECALL_RERANK gates it: ``off`` (default) never calls the
        reranker; ``shadow`` scores and logs the rank delta but returns the
        candidate order untouched (measure p95 before flipping); ``on`` replaces the
        vector score in the relevance blend with the rerank score. The call
        is inline but capped by COLONYMIND_RECALL_RERANK_TIMEOUT_MS
        (default 1200) and, within a turn, by what the host's wait leaves
        (``turn_budget``). It is started only on an idle reranker and never
        queued (``rerank_gate``); an answer after the deadline is used while the
        turn's other lanes are still being assembled. On timeout, a busy
        reranker or an error, this method keeps the candidate order
        and marks selection unavailable; context selection interleaves canonical
        lexical and hybrid order when available. A candidate set that already fits
        skips reranking only when no calibrated abstention cutoff is active.
        A candidate limit bounds model work, while preserving the full input
        for disabled, shadow and failed selection. Past it the candidates sent
        are spread across their sessions (``spread_across_sessions``; a row
        ``last`` marks goes after every other).
        """
        from . import recall_stats, turn_budget
        mode = os.environ.get("COLONYMIND_RECALL_RERANK", "off").strip().lower()
        if mode not in ("shadow", "on"):
            trace.rerank("off", mode=mode)
            return memories, None, False
        budget = turn_budget.current()

        def missed(reason):
            recall_stats.STATS.reranked(turn=budget is not None, missed=reason)
            for memory in memories:
                memory["rerank_status"] = "unavailable"
            trace.rerank("missed", mode=mode, missed=reason)
            return memories, reason, False
        min_score = None
        from colonymind.plasticity.learners.query_adapter import current_request
        request = current_request()
        pinned_cutoff = request.cutoff if request is not None else None
        raw_min_score = (str(pinned_cutoff['cutoff']) if pinned_cutoff is not None else
                         os.environ.get("COLONYMIND_RECALL_RERANK_MIN_SCORE", "").strip())
        if raw_min_score:
            try:
                value = float(raw_min_score)
                if math.isfinite(value):
                    min_score = value
            except (TypeError, ValueError):
                pass
        if min_score is not None:
            from .recall import calibration_fingerprint
            metadata_fn = getattr(self, "_rerank_calibration_metadata", None)
            try:
                metadata = metadata_fn() if metadata_fn is not None else None
                actual = calibration_fingerprint(metadata) if metadata else None
            except Exception:
                metadata, actual = None, None
            expected = (pinned_cutoff['fingerprint'] if pinned_cutoff is not None else
                        os.environ.get("COLONYMIND_RECALL_RERANK_CALIBRATION", "").strip())
            if not actual or not expected or actual != expected:
                status = "mismatch" if expected and actual else "unverified"
                min_score = None
                warning_key = (actual, expected)
                if getattr(self, "_rerank_calibration_warned", None) != warning_key:
                    self._rerank_calibration_warned = warning_key
                    self.logger.warning("Rerank abstention calibration %s; threshold disabled (current configuration: %s)", status, actual or "unknown")
            else:
                status = ("configuration_verified" if metadata.get("weights_revision")
                          and metadata["weights_revision"] != "unverified"
                          else "configuration_verified_weights_unverified")
            for memory in memories:
                memory["rerank_calibration"] = status
            trace.calibration(status, min_score)
        rerank_fn = getattr(self, "_rerank_fn", None)
        if rerank_fn is None and memories:
            return missed("no_reranker")
        if not memories or (len(memories) <= limit and min_score is None):
            trace.rerank("not_needed", mode=mode)
            return memories, None, False
        try:
            timeout_ms = float(os.environ.get(
                "COLONYMIND_RECALL_RERANK_TIMEOUT_MS", "1200"))
        except (TypeError, ValueError):
            timeout_ms = 1200.0
        if not math.isfinite(timeout_ms):
            timeout_ms = 1200.0
        # Within a turn, the deadline is also bounded by what the host's wait leaves (``turn_budget``).
        deadline = max(timeout_ms, 1.0) / 1000.0
        if budget is not None:
            deadline = budget.rerank_timeout(deadline)

        # Combined recall fuses several producers. Overfetch relative to the
        # packet size without making the inline model score every producer's
        # entire result set, and without letting one conversation's chunks take
        # the whole batch. Keep the full list available for fallback.
        submitted = memories if candidate_limit is None else spread_across_sessions(
            memories, max(1, candidate_limit), last=last)
        # Bounded input: an oversized document or query would fail the call and skip the calibrated cutoff.
        docs = [str(m.get("ranking_text", m.get("content", "")))[:RERANK_DOCUMENT_CHARS] for m in submitted]
        # One call at a time per reranker, never queued (``rerank_gate``): a busy reranker is skipped at once.
        # An answer after the deadline is used while the turn's other lanes are still being assembled.
        from .rerank_gate import RerankBusy, rerank_once
        try:
            # The trace records what the reranker is sent when the call starts, whatever comes of it.
            results, late = await rerank_once(
                rerank_fn, str(query)[:RERANK_QUERY_CHARS], docs, timeout_s=deadline,
                late=budget.late_answer if budget is not None else None,
                started=lambda: (recall_exit.reranked(), trace.submitted(submitted, bound=candidate_limit) if trace else None))
        except asyncio.CancelledError as exc:
            recall_exit.failed(exc, "rerank")
            raise
        except Exception as exc:
            if not isinstance(exc, RerankBusy):
                recall_exit.failed(exc, "rerank")
            self._warn_rerank_failure(exc)
            return missed("busy" if isinstance(exc, RerankBusy) else "budget" if deadline <= 0
                          else "timeout" if isinstance(exc, TimeoutError) else "error")

        scores: Dict[int, float] = {}
        try:
            for r in results or []:
                idx = r.get("index") if isinstance(r, dict) else getattr(r, "index", None)
                score = r.get("score") if isinstance(r, dict) else getattr(r, "score", None)
                if not isinstance(idx, int) or isinstance(idx, bool) or idx in scores:
                    raise ValueError("reranker returned an invalid or duplicate index")
                score = float(score)
                if not 0 <= idx < len(submitted) or not math.isfinite(score):
                    raise ValueError("reranker returned an invalid score")
                scores[idx] = score
            # We requested every submitted document. A partial response cannot
            # mix cross-encoder scores with original rank-fusion/ANN scores, or
            # silently treat missing relevance judgments as abstention.
            if len(scores) != len(submitted):
                raise ValueError("reranker returned incomplete scores")
        except (TypeError, ValueError, OverflowError) as exc:
            recall_exit.failed(exc, "rerank")
            self._warn_rerank_failure(exc)
            return missed("invalid")

        recall_stats.STATS.reranked(turn=budget is not None, late=late)
        trace.rerank("shadow" if mode == "shadow" else "scored", mode=mode, candidates=memories, scores=scores,
                     cutoff=min_score, late=late)
        if mode == "shadow":
            ann_top = [m.get("id") for m in sorted(
                memories, key=lambda m: m.get("relevance", 0),
                reverse=True)][:limit]
            rr_idx = sorted(range(len(submitted)),
                            key=lambda i: scores.get(i, float("-inf")),
                            reverse=True)
            rr_top = [submitted[i].get("id") for i in rr_idx[:limit]]
            moved = sum(1 for a, b in zip(ann_top, rr_top) if a != b)
            self.logger.info(
                "recall rerank shadow: candidates=%d limit=%d "
                "top_overlap=%d/%d positions_changed=%d",
                len(submitted), limit, len(set(ann_top) & set(rr_top)),
                limit, moved)
            return memories, None, False

        # mode == "on": rerank score replaces the vector score in the blend.
        # The unsubmitted tail has no comparable model score and cannot enter
        # a successfully reranked packet.
        memories = submitted
        for i, mem in enumerate(memories):
            effective_confidence = (float(
                mem.get("effective_confidence", mem.get("strength", 1.0)))
                if confidence_weighting else 1.0)
            relevance = scores[i] * effective_confidence
            if strength_ranking:
                relevance *= 0.5 + 0.5 * float(mem.get("strength", 1.0))
            mem["relevance"] = relevance
            mem["rerank_score"] = scores[i]
            mem["rerank_status"] = "scored"
        if min_score is not None:
            # Do not fill the context window with unrelated passages merely
            # because there are fewer candidates than requested. Calibration
            # belongs to the serving model/deployment, not a universal constant.
            memories = [mem for i, mem in enumerate(memories)
                        if i in scores and scores[i] >= min_score]
        return memories, None, min_score is not None

    def _warn_rerank_failure(self, exc: BaseException) -> None:
        """Warn on rerank failure at most once per ~5 minutes (fail-open is
        by design; a dead reranker must not turn every recall into a WARNING
        stream)."""
        now = time.monotonic()
        # Sentinel must be None, not 0.0: time.monotonic() is measured from an
        # arbitrary origin (system boot on Linux), so a 0.0 default suppresses
        # the FIRST warning entirely for the first 300s of uptime.
        last = getattr(self, "_rerank_warn_at", None)
        if last is None or now - last >= 300:
            self._rerank_warn_at = now
            self.logger.warning(
                "recall rerank failed (fallback selection): %s: %s", type(exc).__name__, exc)
        else:
            self.logger.debug(
                "recall rerank failed (fallback selection): %s: %s", type(exc).__name__, exc)
