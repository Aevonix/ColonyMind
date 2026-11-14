"""A missed rerank is counted every time, and injects only a bounded fallback.

When recall's reranker missed its deadline (or failed), recall fell back to its candidate order, which skips the
calibrated abstention cutoff, and the only sign was a warning at most once per five minutes. Every miss is now
counted by reason (health's ``recall`` block, turns apart from explicit searches). A missed turn injects what it
did before, the fallback's fused order up to the packet's own limit, until a deployment sets a calibrated smaller
bound (COLONYMIND_RECALL_FALLBACK_LIMIT): fewer records trade answers for fewer injections on questions that should
get none, and only a calibration on positives and negatives can set that trade. The candidate format, and with it
the calibration fingerprint, is unchanged.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.memory import recall, recall_stats, turn_budget
from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
from colonymind.memory.selection import RecallSelector
from colonymind.memory.turn_budget import TurnBudget
from recall_route import ANSWER, Reranker, assemble, recall_app


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIMEOUT_MS", "50")
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", raising=False)
    monkeypatch.delenv(recall_stats.FALLBACK_ENV, raising=False)
    recall_stats.STATS.reset()


FERRY = "The ferry to the island leaves Friday at nine."


def quotations(n=10):
    """Lexical-ranked quotations, the way collection hands them to selection. The only answer-bearing one is the
    fourth (``q-3``): a bound of three records drops it, the packet's own limit of five keeps it."""
    return [{"id": f"q-{i}", "kind": "source_quote", "epistemic_state": "quotation", "role": "user",
             "content": FERRY if i == 3 else f"Passage {i} about the harbour timetable.", "relevance": 1 / (60 + i),
             "_lexical_rank": i} for i in range(n)]


async def slow(query, documents, top_k=10):
    await asyncio.sleep(0.5)
    return [{"index": i, "score": 0.9} for i in range(len(documents))]


async def instant(query, documents, top_k=10):
    return [{"index": i, "score": 0.9 - i / 100} for i in range(len(documents))]


async def select(rerank_fn, *, turn=True, selector=None):
    selector = selector or RecallSelector(rerank_fn)
    if not turn:
        return await selector.select_context("harbour timetable", [], quotations(), limit=5)
    budget = TurnBudget(wait_s=7.5)
    budget.lanes_assembled()  # the turn's other lanes are done: a late answer is not waited for
    with turn_budget.turn(budget):
        return await selector.select_context("harbour timetable", [], quotations(), limit=5)


@pytest.mark.asyncio
async def test_a_missed_turn_keeps_the_answer_the_packet_held_before():
    """No calibrated bound: a missed turn injects the fallback's fused order up to the packet's own limit, as
    recall did before, so the answer in fourth place is still in the context."""
    selected, content = await select(slow)
    assert [row["id"] for row in selected] == ["q-0", "q-1", "q-2", "q-3", "q-4"]
    assert FERRY in content and all(row["rerank_status"] == "unavailable" for row in selected)
    assert content.count('"rerank_status": "unavailable"') == 5
    assert recall_stats.STATS.snapshot()["fallback_limit"] == recall.MEMORY_PACKET_LIMIT == 5


@pytest.mark.asyncio
async def test_a_smaller_bound_is_a_deployments_calibrated_choice(monkeypatch):
    """A deployment that calibrated a smaller bound gets it, and pays for it in answers: at three records the
    answer in fourth place is gone. That is why the bound is not lowered without a calibration."""
    monkeypatch.setenv(recall_stats.FALLBACK_ENV, "3")
    selected, content = await select(slow)
    assert [row["id"] for row in selected] == ["q-0", "q-1", "q-2"] and FERRY not in content
    snapshot = recall_stats.STATS.snapshot()
    assert snapshot["fallback_limit"] == 3 and snapshot["fallback_limit_set"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("value, count", [("1", 1), ("0", 0), ("9", 5), ("many", 5), ("-2", 0), ("", 5)])
async def test_the_fallback_bound_only_ever_lowers_the_packets_limit(monkeypatch, value, count):
    monkeypatch.setenv(recall_stats.FALLBACK_ENV, value)
    selected, content = await select(slow)
    assert len(selected) == count and (content == "") is (count == 0)


@pytest.mark.asyncio
async def test_an_explicit_search_keeps_its_own_limit_on_a_miss():
    """A search that asks for more than a turn's packet got that many records on a miss before, and still does."""
    selected, _ = await RecallSelector(slow).select_context("harbour timetable", [], quotations(), limit=8)
    assert len(selected) == 8


@pytest.mark.asyncio
async def test_a_scored_turn_and_a_turn_without_reranking_keep_the_full_packet(monkeypatch):
    selected, _ = await select(instant)
    assert len(selected) == 5 and all(row["rerank_status"] == "scored" for row in selected)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")  # no reranking configured: not a miss
    selected, _ = await select(slow)
    assert len(selected) == 5
    snapshot = recall_stats.STATS.snapshot()
    assert snapshot["turns"]["scored"] == 1 and snapshot["turns"]["missed"] == 0


@pytest.mark.asyncio
async def test_every_miss_is_counted_by_reason_not_rate_limited(caplog):
    from colonymind.memory.rerank_gate import gate_for
    selector = RecallSelector(slow)  # one selector per reranker, as the sidecar keeps
    with caplog.at_level(logging.WARNING, logger="colonymind.memory.selection"):
        for _ in range(4):
            await select(slow, selector=selector)
            await gate_for(slow).idle()
        blocker = asyncio.ensure_future(select(slow, selector=selector))  # holds the reranker ...
        await asyncio.sleep(0.01)
        await select(slow, selector=selector)  # ... so this one finds it busy
        await blocker
        await gate_for(slow).idle()
    await select(instant)
    await select(slow, turn=False, selector=selector)  # an explicit search is counted apart from turns
    snapshot = recall_stats.STATS.snapshot()
    turns = snapshot["turns"]
    assert turns["missed"] == 6 and turns["missed_by_reason"] == {"timeout": 5, "busy": 1}
    assert turns["scored"] == 1 and turns["reranked"] == 7 and turns["fallback_rows"] == 30
    assert snapshot["searches"]["missed"] == 1 and snapshot["fallback_limit"] == 5
    assert snapshot["fallback_limit_set"] is False
    assert caplog.text.count("recall rerank failed") == 1  # the warning stays rate-limited; the count does not


@pytest.mark.asyncio
async def test_a_late_answer_is_counted_as_scored_late():
    budget = TurnBudget(wait_s=7.5)

    async def answer(query, documents, top_k=10):
        await asyncio.sleep(0.15)
        return [{"index": i, "score": 0.9} for i in range(len(documents))]
    with turn_budget.turn(budget):
        selected, _ = await RecallSelector(answer).select_context("harbour timetable", [], quotations(), limit=5)
    assert len(selected) == 5
    turns = recall_stats.STATS.snapshot()["turns"]
    assert turns["scored"] == 1 and turns["scored_late"] == 1 and turns["missed"] == 0


def test_the_candidate_format_and_its_calibration_fingerprint_are_unchanged(monkeypatch):
    """A calibrated abstention cutoff is tied to this format; a change would silently disable it."""
    for name in ("COLONYMIND_RERANKER_REVISION", "COLONYMIND_EMBED_MODEL", "COLONYMIND_EMBED_DIMS",
                 "COLONYMIND_RECALL_INDEX_GENERATION"):
        monkeypatch.delenv(name, raising=False)
    from colonymind.vector.reranker import OpenAIAPIRerankerProvider
    provider = OpenAIAPIRerankerProvider("fixture-reranker")
    provider.configure("http://reranker.test:8000", "")
    assert recall.CANDIDATE_FORMAT == "grounded-quotation-bundles-v2-corrections-first-rerank-cut-1500-2000"
    assert (recall.RERANK_DOCUMENT_CHARS, recall.RERANK_QUERY_CHARS) == (1500, 2000)
    assert calibration_fingerprint(provider_calibration_metadata(provider)) == PINNED_FINGERPRINT


# The fixture provider's fingerprint on the release this change was built on (no weights revision, embedding or
# index generation set).
PINNED_FINGERPRINT = "6a3b3f04d569b15f7ddc576597ca7aaf82c750ebdf39d06f0716f36b5046910a"


@pytest.mark.asyncio
async def test_on_the_route_the_miss_count_matches_the_turns_that_show_a_fallback(monkeypatch, tmp_path):
    """While every missed turn injects a reranked candidate, as here, the per-turn count equals what a probe sees:
    the turns whose memory carries rerank_status unavailable. Each of those injects at most the packet's limit, and
    still carries the answer in fourth place."""
    reranker = Reranker(delay=0.4)
    setup = await recall_app(monkeypatch, tmp_path, reranker=reranker, rerank_ms=100, initiatives=0, distractors=3)
    from colonymind.memory.rerank_gate import gate_for
    shown = 0
    for delay in (0.4, 0.4, 0.01, 0.4, 0.01):
        reranker.delay = delay
        sections, _ = await assemble(setup.app)
        await gate_for(reranker.rerank).idle()
        memory = sections["colonymind-memory"]["body"]
        if '"rerank_status": "unavailable"' in memory:
            shown += 1
            assert memory.count('"rerank_status": "unavailable"') <= recall.MEMORY_PACKET_LIMIT
        assert ANSWER in memory
    async with AsyncClient(transport=ASGITransport(app=setup.app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        health = (await client.get("/v1/host/health")).json()
    assert health["recall"]["turns"]["missed"] == shown == 3
    assert health["recall"]["turns"]["scored"] == 2


def locators(n):
    """Exact media locators, the way collection hands them to selection: they come before the candidates and are
    never reranked."""
    return [{"id": f"locator-{i}", "kind": "media_locator", "epistemic_state": "retained_attachment_locator",
             "role": "user", "content": "Named retained original.", "matched_names": [f"photo-{i}.jpg"],
             "source_read": {"source_id": f"s-{i}", "view": "image"}, "attachment_index": 0,
             "mime_type": "image/jpeg", "caption_status": "pending", "matching_attachment_candidates": 1,
             "atomic_evidence": True} for i in range(n)]


@pytest.mark.asyncio
@pytest.mark.parametrize("bound, located", [("0", 0), ("", 5)])
async def test_a_miss_whose_fallback_injects_no_candidate_is_counted_and_shows_no_marker(monkeypatch, bound, located):
    """The count does not depend on what a miss injected, so a probe that counts rerank_status markers sees fewer
    misses than the count whenever a missed turn injects no reranked candidate: a bound of 0 injects nothing, and
    exact media locators can fill the packet."""
    monkeypatch.setenv(recall_stats.FALLBACK_ENV, bound)
    budget = TurnBudget(wait_s=7.5)
    budget.lanes_assembled()
    with turn_budget.turn(budget):
        selected, content = await RecallSelector(slow).select_context(
            "harbour timetable", [], locators(located) + quotations(), limit=5)
    assert [row["id"] for row in selected] == [f"locator-{i}" for i in range(located)]
    assert '"rerank_status": "unavailable"' not in content
    assert recall_stats.STATS.snapshot()["turns"]["missed"] == 1
