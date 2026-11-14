"""Remote model providers keep one pooled HTTP client and reuse its connection across calls.

A client opened and closed around every call paid a new connection, and its setup time, on every rerank and query
embedding. Each provider now keeps one client per event loop, and its idle connections expire after
COLONYMIND_HTTP_KEEPALIVE_MS (default
4,000 ms), before a typical server closes them itself.
"""
from __future__ import annotations

import asyncio
import gc
import socket
import time
import warnings
import weakref

import pytest

from model_endpoint import ModelEndpoint, embedding_answer, rerank_answer
from colonymind.vector import http_client
from colonymind.vector.config import EmbeddingConfig
from colonymind.vector.openai_provider import OpenAIAPIEmbeddingProvider
from colonymind.vector.reranker import OpenAIAPIRerankerProvider


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(http_client.KEEPALIVE_ENV, raising=False)


def reranker(url):
    provider = OpenAIAPIRerankerProvider("fixture-reranker")
    provider.configure(url, "")
    return provider


def embedder(url):
    provider = OpenAIAPIEmbeddingProvider(EmbeddingConfig(provider="openai_api", model_id="fixture", dimensions=2))
    provider.configure(url, "")
    return provider


@pytest.mark.asyncio
async def test_a_turns_rerank_calls_share_one_connection():
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)
        for _ in range(3):
            results = await provider.rerank("where is the key", ["blue tin", "red box"], top_k=2)
            assert [row.index for row in results] == [0, 1]
        assert len(endpoint.requests) == 3
        assert endpoint.connections == 1


@pytest.mark.asyncio
async def test_query_embeddings_and_batches_share_one_connection():
    with ModelEndpoint(embedding_answer) as endpoint:
        provider = embedder(endpoint.url)
        assert await provider.embed("a question") == [1.0, 0.0]
        assert await provider.embed_batch(["one", "two"]) == [[1.0, 0.0], [1.0, 0.0]]
        assert await provider.embed("another question") == [1.0, 0.0]
        assert len(endpoint.requests) == 3
        assert endpoint.connections == 1


@pytest.mark.asyncio
async def test_an_idle_connection_expires_before_it_is_reused(monkeypatch):
    """Past the keep-alive expiry the pool opens a new connection instead of offering one the server may have
    closed in the meantime."""
    monkeypatch.setenv(http_client.KEEPALIVE_ENV, "150")
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)
        await provider.rerank("q", ["d"], top_k=1)
        await provider.rerank("q", ["d"], top_k=1)
        assert endpoint.connections == 1
        await asyncio.sleep(0.4)
        await provider.rerank("q", ["d"], top_k=1)
        assert endpoint.connections == 2


def test_the_default_expiry_is_shorter_than_a_typical_server_keep_alive(monkeypatch):
    assert http_client.keepalive_expiry_s() == 4.0 < 5.0
    for bad in ("nan", "-1", "soon"):
        monkeypatch.setenv(http_client.KEEPALIVE_ENV, bad)
        assert http_client.keepalive_expiry_s() == 4.0


def test_a_provider_works_from_a_second_event_loop():
    """A connection belongs to the loop that opened it: a provider used again from a new loop (a worker thread, a
    fresh asyncio.run) gets a client of its own there instead of failing on the ended loop's pool."""
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)
        for _ in range(2):
            results = asyncio.run(provider.rerank("q", ["a", "b"], top_k=2))
            assert len(results) == 2
        assert len(endpoint.requests) == 2


@pytest.mark.asyncio
async def test_reusing_the_connection_saves_its_setup_on_every_call():
    """What the user waits for: on an endpoint that answers at once, a call on the kept connection skips the client
    and connection setup a call paid when it opened its own."""
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)
        await provider.rerank("warm", ["d"], top_k=1)
        started = time.perf_counter()
        for _ in range(20):
            await provider.rerank("q", ["d"], top_k=1)
        pooled = (time.perf_counter() - started) / 20
        assert endpoint.connections == 1
        started = time.perf_counter()
        for _ in range(20):
            await provider.close()  # what every call did before: a client and a connection of its own
            await provider.rerank("q", ["d"], top_k=1)
        fresh = (time.perf_counter() - started) / 20
        assert endpoint.connections == 21 and pooled * 2 < fresh


@pytest.mark.asyncio
async def test_closing_a_provider_closes_its_pooled_connections():
    with ModelEndpoint(embedding_answer) as endpoint:
        provider = embedder(endpoint.url)
        await provider.embed("q")
        await provider.close()
        assert endpoint.idle()  # the socket is closed, not only set aside
        await provider.embed("q")
        assert endpoint.connections == 2
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)
        await provider.rerank("q", ["d"], top_k=1)
        await provider.close()
        assert endpoint.idle()
        await provider.rerank("q", ["d"], top_k=1)
        assert endpoint.connections == 2


@pytest.mark.asyncio
async def test_a_provider_dropped_without_closing_is_closed_on_its_loop_in_order():
    """A provider dropped while its loop runs, without ``close()``: its pool is closed on that loop, in order, and
    the endpoint sees the connection closed. The garbage collector never gets the pooled connection, which it would
    close at once while the loop still counted the socket's descriptor number as the connection's."""
    # Collect earlier tests' unreachable stores before observing this provider's
    # finalizer. SQLite connections warn on collection starting with Python 3.13.
    gc.collect()
    with ModelEndpoint(embedding_answer) as endpoint:
        dropped = embedder(endpoint.url)
        await dropped.embed("q")                  # its connection stays in the pool
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            del dropped
            gc.collect()
            await asyncio.sleep(0.05)
        assert [str(w.message) for w in caught if "unclosed" in str(w.message)] == []
        assert endpoint.idle()


@pytest.mark.asyncio
async def test_a_dropped_providers_pool_never_withdraws_another_callers_watch():
    """What the collector did instead: it closed the pooled socket at once and left the orderly close for later,
    which then withdrew the loop's watch on that descriptor number, by then another caller's. A model call whose new
    connection had the number waited out its whole time limit (the load-dependent connect timeout); here the other
    caller is a socket pair the loop watches, and it still hears its byte."""
    loop = asyncio.get_running_loop()
    with ModelEndpoint(embedding_answer) as endpoint:
        dropped = embedder(endpoint.url)
        await dropped.embed("q")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            del dropped
            gc.collect()
        mine, other = socket.socketpair()
        with mine, other:
            heard = asyncio.Event()
            loop.add_reader(mine.fileno(), heard.set)
            try:
                for _ in range(20):               # the dropped pool's close runs meanwhile
                    await asyncio.sleep(0)
                other.send(b"x")
                await asyncio.wait_for(heard.wait(), 2)
            finally:
                loop.remove_reader(mine.fileno())


def test_an_ended_loop_closes_its_pool_within_it_and_is_released():
    """A provider outlives event loops (a fresh asyncio.run, a worker thread's loop), and a connection belongs to
    the loop that opened it. When a loop ends, its pool is closed within it: the endpoint sees the connection
    closed, the provider keeps no pool for the ended loop, and the loop itself can be collected."""
    loops = []
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)

        async def call():
            loops.append(weakref.ref(asyncio.get_running_loop()))
            assert len(await provider.rerank("q", ["a", "b"], top_k=2)) == 2
        for _ in range(3):
            asyncio.run(call())
            assert endpoint.idle(), "the ended loop's connection is still open"
        gc.collect()
        assert endpoint.connections == 3 and not provider._http._pools
        assert [ref() for ref in loops] == [None, None, None]


def test_a_loop_closed_without_an_orderly_shutdown_is_released_on_the_next_use():
    """A loop closed without shutting down its async generators can no longer close its pool. The provider drops
    that pool on its next use from another loop; the ended loop and its connection are then released."""
    with ModelEndpoint(rerank_answer) as endpoint:
        provider = reranker(endpoint.url)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(provider.rerank("q", ["a"], top_k=1))
        loop.close()
        ended = weakref.ref(loop)
        del loop
        asyncio.run(provider.rerank("q", ["a"], top_k=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)  # the collector closes the ended loop's socket
            gc.collect()
        assert ended() is None and not provider._http._pools
        assert endpoint.idle() and endpoint.connections == 2


def test_the_pool_keeps_httpx_default_sizes():
    """Only the idle expiry changes: the pool keeps httpx's default bounds (100 connections, 20 kept idle)."""
    async def pool():
        client = http_client.KeepAliveClient().get()
        try:
            inner = client._transport._pool
            return inner._max_connections, inner._max_keepalive_connections, inner._keepalive_expiry
        finally:
            await client.aclose()
    assert asyncio.run(pool()) == (100, 20, 4.0)


def test_after_the_lifespan_no_reader_reaches_the_closed_pipeline(tmp_path, monkeypatch):
    """The lifespan closes the embedder's provider when it ends, and with it the pipeline recall reads through
    ``vector.get_pipeline`` (and the store beside it): both are cleared then, so nothing later in the process (a
    test, a harness that starts the app again) embeds a query through a closed provider."""
    from fastapi.testclient import TestClient
    from colonymind import vector
    from colonymind.api.routers import host
    from colonymind.server import create_app
    for name in ("_store", "_pipeline"):
        monkeypatch.setattr(vector, name, getattr(vector, name))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    with ModelEndpoint(embedding_answer) as endpoint:
        for name, value in (("PROVIDER", "openai_api"), ("MODEL", "fixture"), ("DIMS", "2"), ("BASE_URL", endpoint.url)):
            monkeypatch.setenv(f"COLONYMIND_EMBED_{name}", value)
        with TestClient(create_app()):
            assert vector.get_pipeline() is not None and vector.get_store() is not None
        assert vector.get_pipeline() is None and vector.get_store() is None and host._embedder is None
