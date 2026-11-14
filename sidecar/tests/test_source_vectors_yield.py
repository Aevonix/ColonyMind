"""Background source indexing gives the model endpoint to a live recall.

An embedding batch running beside a turn's recall slows the query embedding and the rerank on a shared accelerator,
and during an indexing backlog it can make a turn's recall miss. Recall now marks itself while it runs; the
source-vector lane starts no batch while it does
(waiting up to a bound), and sends a small batch of 4 instead of 16 when recall is still running after that.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from colonymind.memory import live_recall, search
from colonymind.turns import source_vectors
from test_source_vectors import Pipeline, setup


class Recording(Pipeline):
    """Records when each batch starts and how large it is, and whether a live recall was running then."""

    def __init__(self):
        super().__init__()
        self.batches = []

    async def embed_batch(self, texts):
        self.batches.append({"at": time.monotonic(), "size": len(texts), "recall": live_recall.LIVE.active()})
        return await super().embed_batch(texts)


async def backlog(tmp_path, chunks=40):
    pipeline = Recording()
    ledger, store, _, projection = await setup(tmp_path, pipeline)
    ledger.record_source("long", contact_id="c", session_id="s", messages=[
        {"role": "user", "content": f"Historical item {index}"} for index in range(chunks)])
    projection.backfill()
    return pipeline, projection


@pytest.mark.asyncio
async def test_no_batch_starts_while_a_live_recall_runs(tmp_path):
    pipeline, projection = await backlog(tmp_path)
    with live_recall.LIVE.running():
        lane = asyncio.create_task(projection.process_one())
        await asyncio.sleep(0.3)
        assert pipeline.batches == [] and not lane.done()
    ended = time.monotonic()
    assert await asyncio.wait_for(lane, 5)
    (batch,) = pipeline.batches
    assert batch["at"] >= ended and batch["size"] == 16 and not batch["recall"]


@pytest.mark.asyncio
async def test_a_long_recall_gets_small_batches_instead(tmp_path, monkeypatch):
    """The backlog still drains under constant traffic: after the bound, a batch of 4 instead of 16."""
    monkeypatch.setattr(live_recall, "YIELD_SECONDS", 0.2)
    pipeline, projection = await backlog(tmp_path)
    with live_recall.LIVE.running():
        started = time.monotonic()
        assert await projection.process_one()
        assert time.monotonic() - started >= 0.2
    assert await projection.process_one()
    assert [batch["size"] for batch in pipeline.batches] == [4, 16]


@pytest.mark.asyncio
async def test_recall_is_marked_live_while_it_collects_and_selects(monkeypatch):
    seen = []

    class Vectors:
        def __init__(self, *args):
            pass

        async def search(self, query, **kwargs):
            seen.append(("embedding", live_recall.LIVE.active()))
            return [], []

    class Ledger:
        def erasure_watermark(self, contact_id):
            return 0

        def search_sources(self, query, **kwargs):
            seen.append(("keyword", live_recall.LIVE.active()))
            return []

        def search_stems(self, query, **kwargs):
            seen.append(("stemmed", live_recall.LIVE.active()))
            return []

        def search_claim_keys(self, query, **kwargs):
            seen.append(("claim_key", live_recall.LIVE.active()))
            return []

    class Selector:
        async def select_context(self, query, beliefs, quotations, **kwargs):
            seen.append(("rerank", live_recall.LIVE.active()))
            return [], ""
    monkeypatch.setattr(source_vectors, "SourceVectors", Vectors)
    collected = await search.collect_sources(Ledger(), query="q", contact_id="c", session_id="s")
    collected.ledger = None
    await search.select_memory(collected, query="q", selector=Selector(), timezone_name="UTC")
    assert sorted(seen) == [("claim_key", True), ("embedding", True), ("keyword", True), ("rerank", True),
                            ("stemmed", True)]
    assert not live_recall.LIVE.active()


@pytest.mark.asyncio
async def test_a_turns_recall_during_a_backlog_never_overlaps_a_new_batch(tmp_path, monkeypatch):
    """The lane runs continuously while three turns recall: no batch starts inside a recall."""
    pipeline, projection = await backlog(tmp_path, chunks=400)
    progress = asyncio.Queue()

    async def lane():
        while await projection.process_one():
            progress.put_nowait(len(pipeline.batches))
            await asyncio.sleep(0.01)
    indexing = asyncio.create_task(lane())
    try:
        windows = []
        for _ in range(3):
            # Observe a completed batch before each turn, regardless of runner throughput.
            await asyncio.wait_for(progress.get(), 5)
            with live_recall.LIVE.running():
                begun = time.monotonic()
                await asyncio.sleep(0.15)  # the recall's model calls
                windows.append((begun, time.monotonic()))
        assert len(pipeline.batches) >= 3
        # A batch the lane decided on just before a recall began may start within one loop turn of it.
        assert not [b for b in pipeline.batches if any(start + 0.02 < b["at"] <= end for start, end in windows)]
    finally:
        indexing.cancel()
        await asyncio.gather(indexing, return_exceptions=True)
