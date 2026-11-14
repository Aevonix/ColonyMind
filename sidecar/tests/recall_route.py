"""The context route with real recall behind it: a source ledger, a Lance source index, a controlled embedder and a
controlled reranker. Shared by the recall latency tests."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind import vector
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.memory import compass
from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.source_vectors import SourceVectors
from colonymind.vector.indexes import EmbeddingIdentity, IndexCatalog
from colonymind.vector.store import VectorStore

OWNER = "contact-a"
ANSWER = "The hydrofoil departure is Friday at nine."


class Embedder:
    """Indexes at once; a live query takes ``query_s`` (None: never answers)."""

    def __init__(self, query_s: float | None = 0.0):
        self.index_identity = EmbeddingIdentity("fixture", "fixture", "unknown", 3)
        self.query_s, self.queries = query_s, []

    @staticmethod
    def vector(text):
        return [1.0, 0.0, 0.0] if "hydrofoil" in text.casefold() else [0.0, 1.0, 0.0]

    async def embed_batch(self, texts):
        return [self.vector(text) for text in texts]

    async def embed_query(self, text):
        started = time.monotonic()
        self.queries.append({"started": started, "ended": None})
        try:
            await asyncio.sleep(3600 if self.query_s is None else self.query_s)
            return self.vector(text)
        finally:
            self.queries[-1]["ended"] = time.monotonic()


class Reranker:
    """Answers after ``delay`` seconds (None: never within a test); scores the answer high. ``capacity_one``
    serves one call at a time, like a server that finishes an abandoned call before the next."""

    def __init__(self, delay: float | None = 0.0, *, capacity_one: bool = False):
        self.delay, self.calls = delay, []
        self.lock = asyncio.Lock() if capacity_one else None

    def calibration_metadata(self):
        return {"model": "route-fixture", "format_version": "v1"}

    async def _answer(self, query, documents):
        self.calls.append({"started": time.monotonic(), "documents": len(documents),
                           "judge": any(doc.startswith("Pending Initiatives: ") for doc in documents)})
        await asyncio.sleep(3600 if self.delay is None else self.delay)
        return [{"index": i, "score": 0.99 if "Friday at nine" in doc else 0.1} for i, doc in enumerate(documents)]

    async def rerank(self, query, documents, top_k=10):
        if self.lock is None:
            return await self._answer(query, documents)
        async with self.lock:
            return await self._answer(query, documents)


async def recall_app(monkeypatch, tmp_path, *, embedder=None, reranker=None, rerank_ms=2500, judge_ms=1000,
                     initiatives=5, distractors=7):
    """The app, the ledger and the index, with the answer and ``distractors`` distractors recorded and projected.
    Every distractor matches the query's words more often than the answer, so without a reranker's scores the answer
    comes after all of them: with three it is the fourth record of the fallback's order. ``initiatives`` pending
    initiatives give the per-turn judge optional items to score (``assemble`` asks for them)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    for name in ("_contacts_store", "_telemetry", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    embedder = embedder or Embedder()
    reranker = reranker or Reranker()
    monkeypatch.setattr(host, "_reranker", reranker)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIMEOUT_MS", str(rerank_ms))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", ".8")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION",
                       calibration_fingerprint(provider_calibration_metadata(reranker)))
    monkeypatch.setenv(compass.TIMEOUT_ENV, str(judge_ms))
    monkeypatch.setenv(compass.BUDGET_ENV, "200")
    initiative = SimpleNamespace(type="garden", priority=0.5,
                                 description="Plant the spring bulbs along the east fence. " * 3)
    monkeypatch.setattr(host, "_initiative_store",
                        SimpleNamespace(list=lambda status, limit: [initiative] * initiatives))
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("answer", contact_id=OWNER, session_id="session-a",
                         messages=[{"role": "user", "content": "ordinary context " * 20 + ANSWER}])
    for n in range(distractors):
        ledger.record_source(f"d{n}", contact_id=OWNER, session_id="session-a", messages=[
            {"role": "user", "content": f"Hydrofoil departure board {n}: hydrofoil departure listings."}])
    store = VectorStore(str(tmp_path / "lancedb"), identity=embedder.index_identity, catalog=IndexCatalog(ledger))
    await store.connect(3)
    await store.ensure_collections(3)
    projection = SourceVectors(ledger, store, embedder)
    projection.backfill()
    for _ in range(50):
        if not await projection.process_one():
            break
    monkeypatch.setattr(vector, "_store", store)
    monkeypatch.setattr(vector, "_pipeline", embedder)
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    app.include_router(host.router)
    return SimpleNamespace(app=app, ledger=ledger, store=store, embedder=embedder, reranker=reranker)


async def assemble(app, query="hydrofoil departure", **extra):
    """POST /context/assemble; returns (sections by id, seconds)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        started = time.monotonic()
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "fixture"}, "context": {"contact_id": OWNER, "session_id": "session-b"},
            "incoming_message": {"role": "user", "content": query}, "include_initiatives": True, **extra})
        elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    return {section["id"]: section for section in response.json()["sections"]}, elapsed


async def search(app, query="hydrofoil departure"):
    """POST /memory/search (an explicit search: no turn budget); returns (body, seconds)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        started = time.monotonic()
        response = await client.post("/v1/host/memory/search", json={
            "identity": {"host_id": "fixture"}, "person_id": OWNER, "session_id": "session-b", "query": query})
        elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    return response.json(), elapsed
