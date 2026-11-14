"""A deployment's calibrated cutoff is bound, by its fingerprint, to what recall sends the reranker.

The cutoff applies only while ``COLONYMIND_RECALL_RERANK_CALIBRATION`` matches the running configuration's
fingerprint. A change to the candidate format, or to what the fingerprint covers, switches every deployment's cutoff
off (the mismatch fallback) until it is recalibrated, and without the cutoff recall injects every candidate. Such a
change is deliberate and comes with the calibration docs (docs/RECALL-HYBRID.md) that tell an operator what the
format is now: these tests fail when the code and those docs disagree, and they pin the format deployments are
calibrated on. (The ranking texts themselves are pinned where they are made: test_source_annotations for
corrections first, test_source_vectors for a conversation pair.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from colonymind.memory import recall
from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
from colonymind.memory.selection import RecallSelector
from colonymind.vector.reranker import OpenAIAPIRerankerProvider

DOCS = Path(__file__).resolve().parents[2] / "docs" / "RECALL-HYBRID.md"

# The calibrated format, and the reference fingerprints of the fixture configuration below, as the first Proto-AGI
# release computed them. They move only with a deliberate format change, which the docs and every deployment's
# calibration follow.
CALIBRATED_FORMAT = "grounded-quotation-bundles-v2-corrections-first-rerank-cut-1500-2000"
REFERENCE_FINGERPRINTS = {
    "": "6a3b3f04d569b15f7ddc576597ca7aaf82c750ebdf39d06f0716f36b5046910a",
    "qwen3": "4ec2ca0198780bb21f3f7af6412d6ac21823360beaef3b507145a6c5af72bdc3",
}
ENVIRONMENT_INPUTS = ("COLONYMIND_RERANKER_REVISION", "COLONYMIND_EMBED_MODEL", "COLONYMIND_EMBED_DIMS",
                      "COLONYMIND_RECALL_INDEX_GENERATION")


def fixture_metadata(monkeypatch, prompt_style=""):
    """The docs' fixture configuration: no key, and none of the environment inputs set."""
    for name in ENVIRONMENT_INPUTS:
        monkeypatch.delenv(name, raising=False)
    provider = OpenAIAPIRerankerProvider("fixture-reranker")
    provider.configure("http://reranker.test:8000", "", prompt_style)
    return provider_calibration_metadata(provider)


def docs():
    """The calibration docs as prose: line breaks are only wrapping."""
    return " ".join(DOCS.read_text().split())


def test_the_candidate_format_is_the_calibrated_one_and_the_docs_name_it():
    assert recall.CANDIDATE_FORMAT == CALIBRATED_FORMAT
    assert (recall.RERANK_DOCUMENT_CHARS, recall.RERANK_QUERY_CHARS) == (1500, 2000)
    text = docs()
    assert f"The calibrated candidate format is `{recall.CANDIDATE_FORMAT}`." in text
    assert set(re.findall(r"`(grounded-quotation-bundles[^`]*)`", text)) == {recall.CANDIDATE_FORMAT}
    assert (f"the first {recall.RERANK_DOCUMENT_CHARS:,} characters of each candidate's ranking text and the first "
            f"{recall.RERANK_QUERY_CHARS:,} characters of the query") in text


def test_the_docs_name_every_fingerprint_input(monkeypatch):
    listed = re.search(r"For a registered reranker its inputs, by name, are (.+?)\. ", docs())
    assert listed, "docs/RECALL-HYBRID.md no longer lists the fingerprint's inputs"
    metadata = fixture_metadata(monkeypatch)
    assert sorted(re.findall(r"`([a-z_]+)`", listed.group(1))) == sorted(metadata)
    assert metadata["candidate_format"] == CALIBRATED_FORMAT
    # Each environment input the docs' fixture leaves unset is an input of its own.
    for name in ENVIRONMENT_INPUTS:
        assert f"`{name}`" in docs()


@pytest.mark.parametrize("prompt_style", ["", "qwen3"])
def test_the_reference_fingerprints_are_the_calibrated_ones_and_the_docs_publish_them(monkeypatch, prompt_style):
    fingerprint = calibration_fingerprint(fixture_metadata(monkeypatch, prompt_style))
    assert fingerprint == REFERENCE_FINGERPRINTS[prompt_style]
    assert f"`{fingerprint}`" in docs()


@pytest.mark.asyncio
async def test_recall_sends_one_cut_window_per_candidate_and_the_cut_query(monkeypatch):
    sent = []

    class Reranker:
        def calibration_metadata(self):
            return {"model": "neutral-fixture", "format_version": "v1"}

        async def rerank(self, query, documents, top_k):
            sent.append((query, list(documents)))
            return [{"index": i, "score": .9} for i in range(len(documents))]

    reranker = Reranker()
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", ".5")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", calibration_fingerprint(
        provider_calibration_metadata(reranker)))
    selector = RecallSelector(reranker.rerank, calibration_metadata=lambda: provider_calibration_metadata(reranker))
    ranked_text = "User input:\n" + "a" * 1600 + "\nAssistant response:\nb"
    candidates = [{"id": "pair", "kind": "conversation_pair", "content": "{}", "ranking_text": ranked_text,
                   "relevance": .5},
                  {"id": "quote", "kind": "source_quote", "content": "c" * 1700, "relevance": .4}]
    query = "q" * 2100
    await selector.rerank(query, candidates, 5, confidence_weighting=False, candidate_limit=20)
    assert sent == [(query[:2000], [ranked_text[:1500], "c" * 1500])]
