"""The query embedding path: no retry the endpoint will refuse again, a query cut below the model's context, a small
cache of its own, and an embedding that runs while the keyword search does.

An HTTP 400 (a query longer than the embedding model's context) was retried three times with pauses before recall
gave up on it; the embedding waited for the keyword search to finish before it started; and the pipeline's one cache
kept every background batch while live queries, which carry changing context, rarely hit it.
"""
from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from model_endpoint import ModelEndpoint, embedding_answer
from colonymind.memory import search
from colonymind.turns import source_vectors
from colonymind.vector import embedder as embedder_module
from colonymind.vector.config import EmbeddingConfig
from colonymind.vector.embedder import EmbeddingPipeline
from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider


def provider(url):
    p = OpenAIAPIEmbeddingProvider(EmbeddingConfig(provider="openai_api", model_id="fixture", dimensions=2))
    p.configure(url, "")
    return p


def status(code):
    return lambda path, body: (code, {"error": {"message": "fixture %d" % code}})


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [400, 413, 422])
async def test_a_request_the_endpoint_refuses_is_not_retried(code):
    """A 4xx is the endpoint refusing this request (too long, malformed): the same request is refused again."""
    with ModelEndpoint(status(code)) as endpoint:
        started = time.perf_counter()
        with pytest.raises(httpx.HTTPStatusError):
            await provider(endpoint.url).embed_batch(["a source chunk"])
        assert len(endpoint.requests) == 1 and time.perf_counter() - started < 0.4


@pytest.mark.asyncio
async def test_a_transient_failure_is_still_retried_for_background_batches():
    with ModelEndpoint(status(503)) as endpoint:
        with pytest.raises(httpx.HTTPStatusError):
            await provider(endpoint.url).embed_batch(["a source chunk"])
        assert len(endpoint.requests) == 3


@pytest.mark.asyncio
async def test_a_query_embedding_is_one_attempt():
    """A turn cannot wait for retries: recall keeps its keyword evidence and goes on."""
    with ModelEndpoint(status(503)) as endpoint:
        pipeline = EmbeddingPipeline(provider(endpoint.url))
        started = time.perf_counter()
        with pytest.raises(httpx.HTTPStatusError):
            await pipeline.embed_query("where is the key")
        assert len(endpoint.requests) == 1 and time.perf_counter() - started < 0.4


def sent_bound(tokens):
    """The most UTF-8 bytes of instruction and query that ``tokens`` admit: a tokenizer emits at most one token per
    byte of its input, plus a few markers of its own."""
    return (tokens - embedder_module.QUERY_SPECIAL_TOKENS) * embedder_module.BYTES_PER_TOKEN_FLOOR


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", ["pasted log line ", "\u00e9", "\u6f22", "\U00020000", "a\U0001f600"])
async def test_a_long_query_is_cut_to_what_the_token_limit_admits_in_any_script(monkeypatch, unit):
    """The instruction and the query together are cut to the token limit counted in UTF-8 bytes, so no tokenizer
    can take them past it: text of four-byte characters keeps a quarter of the characters ASCII text keeps. The cut
    keeps as much as fits and never splits a character."""
    monkeypatch.delenv(embedder_module.QUERY_TOKENS_ENV, raising=False)
    with ModelEndpoint(embedding_answer) as endpoint:
        pipeline = EmbeddingPipeline(provider(endpoint.url))
        query = "Where is the key? " + unit * 20000
        await pipeline.embed_query(query)
        (_, body), = endpoint.requests
        sent, = body["input"]
        size = len(sent.encode("utf-8"))
        assert sent.startswith(pipeline.query_instruction) and query.startswith(sent[len(pipeline.query_instruction):])
        assert sent_bound(embedder_module.DEFAULT_QUERY_TOKENS) - 4 < size <= sent_bound(
            embedder_module.DEFAULT_QUERY_TOKENS)


@pytest.mark.asyncio
async def test_the_token_limit_is_a_setting_and_a_short_query_is_sent_whole(monkeypatch):
    monkeypatch.setenv(embedder_module.QUERY_TOKENS_ENV, "512")
    with ModelEndpoint(embedding_answer) as endpoint:
        pipeline = EmbeddingPipeline(provider(endpoint.url))
        await pipeline.embed_query("\U00020000" * 1000)
        sent, = endpoint.requests[-1][1]["input"]
        assert sent_bound(512) - 4 < len(sent.encode("utf-8")) <= sent_bound(512)
        await pipeline.embed_query("where is the blue tin")
        assert endpoint.requests[-1][1]["input"] == [pipeline.query_instruction + "where is the blue tin"]


class Counting:
    """An in-process provider that counts what it embeds."""

    def __init__(self):
        self.single, self.batches = [], []
        self._config = EmbeddingConfig(provider="cpu", model_id="fixture", dimensions=2)

    @property
    def dimensions(self):
        return 2

    async def embed(self, text):
        self.single.append(text)
        return [1.0, float(len(text))]

    async def embed_batch(self, texts):
        self.batches.append(list(texts))
        return [[1.0, float(len(text))] for text in texts]


@pytest.mark.asyncio
async def test_queries_have_a_small_lru_cache_of_their_own(monkeypatch):
    monkeypatch.setenv(embedder_module.QUERY_CACHE_ENV, "2")
    counting = Counting()
    pipeline = EmbeddingPipeline(counting)
    for query in ("alpha", "beta", "alpha", "gamma", "alpha", "beta"):
        await pipeline.embed_query(query)
    # alpha was used again before gamma arrived, so beta (the least recently used) made room; beta then misses.
    assert [text.removeprefix(pipeline.query_instruction) for text in counting.single] == [
        "alpha", "beta", "gamma", "beta"]
    assert len(pipeline._query_cache) == 2 and not pipeline._cache


@pytest.mark.asyncio
async def test_background_batches_are_not_cached():
    """A source chunk is embedded once; keeping its vector only held memory."""
    counting = Counting()
    pipeline = EmbeddingPipeline(counting)
    await pipeline.embed_batch(["chunk one", "chunk two"])
    await pipeline.embed_batch(["chunk one", "chunk two"])
    assert counting.batches == [["chunk one", "chunk two"]] * 2
    assert not pipeline._cache and not pipeline._query_cache


class Hits:
    """A ledger whose keyword search takes a while (it runs on a worker thread)."""

    def __init__(self, started: threading.Event, *, wait=2.0):
        self.started, self.wait, self.saw_embedding = started, wait, None

    def erasure_watermark(self, contact_id):
        return 0

    def search_sources(self, query, **kwargs):
        self.saw_embedding = self.started.wait(self.wait)
        return [{"turn_id": "t-1", "role": "user", "content": "The key is in the blue tin.",
                 "source_message_hash": "h-1"}]

    def search_stems(self, query, **kwargs):
        return []

    def search_claim_keys(self, query, **kwargs):
        return []


@pytest.mark.asyncio
async def test_the_query_embedding_runs_while_the_keyword_search_does(monkeypatch):
    started = threading.Event()

    class Vectors:
        def __init__(self, *args):
            pass

        async def search(self, query, **kwargs):
            started.set()
            await asyncio.sleep(0.3)
            return [], []
    monkeypatch.setattr(source_vectors, "SourceVectors", Vectors)
    ledger = Hits(started)
    begun = time.perf_counter()
    collected = await search.collect_sources(ledger, query="where is the key", contact_id="c", session_id="s")
    elapsed = time.perf_counter() - begun
    assert ledger.saw_embedding is True
    assert [hit["turn_id"] for hit in collected.hits] == ["t-1"] and collected.lexical_hits == collected.hits
    assert elapsed < 0.9  # the two overlap: not the keyword search's wait plus the embedding's 0.3 s


@pytest.mark.asyncio
async def test_a_cancelled_collection_cancels_and_awaits_its_embedding(monkeypatch):
    """A turn that is cancelled (its caller went away) leaves no embedding running behind it."""
    started, release = threading.Event(), threading.Event()
    state = {"cancelled": False, "finished": False}

    class Vectors:
        def __init__(self, *args):
            pass

        async def search(self, query, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            finally:
                state["finished"] = True
    monkeypatch.setattr(source_vectors, "SourceVectors", Vectors)

    class Blocking(Hits):
        def search_sources(self, query, **kwargs):
            release.wait(5)
            return []
    task = asyncio.create_task(search.collect_sources(Blocking(started), query="q", contact_id="c", session_id="s"))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.02)
        assert started.is_set(), "the embedding did not start while the keyword search ran"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state == {"cancelled": True, "finished": True}
    finally:
        release.set()
