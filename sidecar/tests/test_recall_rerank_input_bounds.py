"""Recall sends the reranker bounded input, so its answer, and the calibrated cutoff, do not depend on how long the
recalled messages are.

A reranker endpoint refuses a request whose document is longer than its context (an HTTP 400), and a very long
batch can outlast recall's deadline. Either failure makes recall fall back to its candidate order, which skips the
calibrated abstention cutoff: every candidate is injected, relevant or not. Measured on a copy of a deployed store,
with recall given 10 s: 37% of questions that should get no memory, and 7% of real memory questions, were never
scored, mostly because one recalled message ran to 11,000 to 35,000 characters. Recall now sends at most
RERANK_DOCUMENT_CHARS of each candidate and RERANK_QUERY_CHARS of the query; the text shown to the model is
unchanged. The cut is part of the calibrated candidate format, so a cutoff calibrated on uncut input is not
applied silently.
"""
from __future__ import annotations

import pytest

from colonymind.memory import recall, selection
from colonymind.memory.selection import RecallSelector

LIMIT = 4000  # what the fake endpoint accepts per document, like a reranker with a bounded context


def memories():
    """Six candidates: five short ones and the owner's own long message, which carries the answer early on."""
    rows = [{"id": f"m-{n}", "kind": "source_quote", "content": f"Note {n} about the garden shed paint.",
             "relevance": 0.5} for n in range(5)]
    owner = "The spare key to the studio is in the blue tin under the stairs. " + "Unrelated diary text. " * 1200
    rows.append({"id": "owner", "kind": "source_quote", "content": owner, "ranking_text": owner, "relevance": 0.4})
    return rows


class BoundedReranker:
    """Refuses an oversized request like the reranker endpoint does; otherwise scores the key question's answer."""

    def __init__(self):
        self.calls = []

    async def __call__(self, query, documents, top_k=10):
        self.calls.append((query, list(documents)))
        if len(query) > LIMIT or any(len(document) > LIMIT for document in documents):
            raise RuntimeError("Client error '400 Bad Request' for url 'http://reranker/v1/rerank'")
        return [{"index": i, "score": 0.95 if "blue tin" in document else 0.05} for i, document in enumerate(documents)]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Where is the spare key to the studio?",
                                   "Where is the spare key to the studio? " + "pasted log line " * 600])
async def test_a_long_recalled_message_or_question_is_still_scored(monkeypatch, query):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", raising=False)
    reranker = BoundedReranker()
    ranked = await RecallSelector(reranker).rerank(query, memories(), 5, confidence_weighting=False,
                                                   candidate_limit=20)
    assert all(row.get("rerank_status") == "scored" for row in ranked)
    assert max(ranked, key=lambda row: row["relevance"])["id"] == "owner"
    ((sent_query, documents),) = reranker.calls
    assert len(sent_query) <= selection.RERANK_QUERY_CHARS and sent_query == query[:selection.RERANK_QUERY_CHARS]
    assert all(len(document) <= selection.RERANK_DOCUMENT_CHARS for document in documents)
    # What the model is shown is not cut.
    assert len(next(row for row in ranked if row["id"] == "owner")["content"]) > selection.RERANK_DOCUMENT_CHARS


def test_the_cut_is_part_of_the_calibrated_candidate_format():
    class Provider:
        def calibration_metadata(self):
            return {"model": "fixture"}
    fmt = recall.provider_calibration_metadata(Provider())["candidate_format"]
    assert fmt != "grounded-quotation-bundles-v2-corrections-first"
    assert str(selection.RERANK_DOCUMENT_CHARS) in fmt and str(selection.RERANK_QUERY_CHARS) in fmt
