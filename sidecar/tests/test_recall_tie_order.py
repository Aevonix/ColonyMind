"""Near-tied rerank scores keep the fused retrieval order.

Once the reranker scores every candidate, the person's own words tie with the agent's restatement of them and with
long tool output that repeats the same facts: all of them answer the question, and the cross-encoder's scores for
them differ by thousandths. Sorting on those thousandths put the restatement first and let the packet's budget drop
the person's words. Scores within a margin of each other are a tie, and the fused retrieval order (lexical and
semantic rank) breaks it. A clear preference is kept, and the scores, the cutoff and the calibrated configuration
stamp are unchanged. What difference is a tie depends on the scale of one deployment's scores, which a verified
cutoff does not establish: the margin is that deployment's setting, and there is none by default.
"""

from __future__ import annotations

import pytest

from colonymind.memory.recall import (
    CANDIDATE_FORMAT, calibration_fingerprint, provider_calibration_metadata, source_candidates,
)
from colonymind.memory import selection
from colonymind.memory.selection import RecallSelector

OWNER = "The quarterly review is on Tuesday at 09:15 in the studio two."
RESTATED = "Noted: quarterly review, Tuesday 09:15, studio two. I will remind you the evening before."
TOOL = "calendar.create returned: quarterly review, Tuesday 09:15, studio two, reminder set for Monday 18:00."
OTHER = "The studio two booking sheet lists the projector as out for repair."


class Reranker:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def calibration_metadata(self):
        return {"model": "neutral-fixture", "format_version": "v1"}

    async def rerank(self, query, documents, top_k):
        self.calls.append(list(documents))
        return [{"index": i, "score": self.scores[text]} for i, text in enumerate(documents)]


MARGIN = ".04"   # a margin this fixture's deployment validated for its score scale


def calibrated(monkeypatch, reranker, *, cutoff=".8", margin=MARGIN):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", cutoff)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", calibration_fingerprint(
        provider_calibration_metadata(reranker)))
    if margin is None:
        monkeypatch.delenv("COLONYMIND_RECALL_RERANK_TIE_MARGIN", raising=False)
    else:
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIE_MARGIN", margin)
    return RecallSelector(reranker.rerank, calibration_metadata=lambda: provider_calibration_metadata(reranker))


def fused(*texts):
    """Source quotations in fused retrieval order (the first is the best lexical and semantic match)."""
    return source_candidates([{"turn_id": f"turn-{i}", "role": role, "content": text, "contact_id": "person",
                               "scope": "person", "source_message_hash": f"{i:064x}"}
                              for i, (role, text) in enumerate(texts)])


async def order(selector, rows):
    selected, content = await selector.select_context("When and where is the quarterly review?", [], rows)
    return [row["content"] for row in selected], selected, content


@pytest.mark.asyncio
async def test_the_persons_words_lead_their_near_tied_restatements(monkeypatch):
    reranker = Reranker({OWNER: .984, RESTATED: .99, TOOL: .995, OTHER: .1})
    selector = calibrated(monkeypatch, reranker)
    texts, selected, content = await order(selector, fused(
        ("user", OWNER), ("assistant", RESTATED), ("tool", TOOL), ("user", OTHER)))
    assert texts == [OWNER, RESTATED, TOOL]                     # a tie keeps the fused order; OTHER is under the cutoff
    assert content.index(OWNER) < content.index("I will remind you")
    # The scores and the calibrated cutoff are untouched: only the order among near ties changed.
    assert [row["rerank_score"] for row in selected] == [.984, .99, .995]
    assert all(row["rerank_status"] == "scored" for row in selected)
    assert {row["rerank_calibration"] for row in selected} == {"configuration_verified_weights_unverified"}


@pytest.mark.asyncio
async def test_a_clear_preference_keeps_the_reranker_order(monkeypatch):
    reranker = Reranker({OTHER: .85, OWNER: .99})
    selector = calibrated(monkeypatch, reranker)
    texts, _, _ = await order(selector, fused(("user", OTHER), ("user", OWNER)))
    assert texts == [OWNER, OTHER]


@pytest.mark.asyncio
async def test_no_candidate_goes_ahead_of_one_that_scored_more_than_the_margin_higher(monkeypatch):
    # In fused order: .93, .96, .99. The best is .99; .96 is within the margin of it, .93 is not.
    first, second, third = "First fused passage.", "Second fused passage.", "Third fused passage."
    reranker = Reranker({first: .93, second: .96, third: .99})
    selector = calibrated(monkeypatch, reranker)
    texts, _, _ = await order(selector, fused(("user", first), ("user", second), ("user", third)))
    assert texts == [second, third, first]
    assert .99 - .93 > float(MARGIN) >= .99 - .96


@pytest.mark.asyncio
@pytest.mark.parametrize("margin,expected", [
    (None, [TOOL, RESTATED, OWNER]),                             # unset: no margin, the reranker's order
    ("0", [TOOL, RESTATED, OWNER]),                              # no margin: the reranker's order
    (".007", [RESTATED, TOOL, OWNER]),                          # restatement and tool output tie; the owner trails by .011
    (".04", [OWNER, RESTATED, TOOL]),
    ("not a number", [TOOL, RESTATED, OWNER]),                  # an unreadable setting keeps the default: none
    ("-1", [TOOL, RESTATED, OWNER]),                            # a negative margin is none
])
async def test_the_margin_is_a_setting(monkeypatch, margin, expected):
    reranker = Reranker({OWNER: .984, RESTATED: .99, TOOL: .995})
    selector = calibrated(monkeypatch, reranker, margin=margin)
    texts, _, _ = await order(selector, fused(("user", OWNER), ("assistant", RESTATED), ("tool", TOOL)))
    assert texts == expected


@pytest.mark.asyncio
async def test_a_verified_cutoff_on_a_narrow_scale_keeps_the_reranker_order(monkeypatch):
    # A deployment whose reranker scores on a narrow scale calibrated its own cutoff (.0005) and set no margin. The
    # fingerprint verifies its configuration, not what size of difference is a tie on its scale: .001 and .03 are a
    # thirtyfold difference there. Five weak candidates come first in fused order; the answer still leads.
    weak = [f"Unrelated passage {i}." for i in range(5)]
    reranker = Reranker({OWNER: .03, **{text: .001 for text in weak}})
    selector = calibrated(monkeypatch, reranker, cutoff=".0005", margin=None)
    texts, selected, _ = await order(selector, fused(*(("user", text) for text in weak), ("user", OWNER)))
    assert texts == [OWNER, *weak[:4]]
    assert {row["rerank_calibration"] for row in selected} == {"configuration_verified_weights_unverified"}
    assert selection.RERANK_TIE_MARGIN == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stamp", ["absent", "mismatch"])
async def test_without_a_verified_calibration_the_reranker_order_stands(monkeypatch, stamp):
    # Scores mean something only on the scale a deployment calibrated; without a verified stamp there is no cutoff,
    # and no margin either (an absolute margin would merge .001 and .01, a tenfold difference).
    fillers = [f"Unrelated passage {i}." for i in range(4)]    # more candidates than the packet holds, so it is scored
    reranker = Reranker({OWNER: .984, RESTATED: .99, TOOL: .995, **{text: .01 for text in fillers}})
    selector = calibrated(monkeypatch, reranker)
    if stamp == "absent":
        monkeypatch.delenv("COLONYMIND_RECALL_RERANK_MIN_SCORE")
    else:
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", "0" * 64)
    texts, selected, _ = await order(selector, fused(
        ("user", OWNER), ("assistant", RESTATED), ("tool", TOOL), *(("user", text) for text in fillers)))
    assert texts[:3] == [TOOL, RESTATED, OWNER]
    assert all(row["rerank_status"] == "scored" for row in selected)


@pytest.mark.asyncio
async def test_an_exact_repeat_of_the_request_still_goes_last(monkeypatch):
    question = "When and where is the quarterly review?"
    reranker = Reranker({question: .999, OWNER: .984})
    selector = calibrated(monkeypatch, reranker)
    texts, _, _ = await order(selector, fused(("user", question), ("user", OWNER)))
    assert texts == [OWNER, question]


def test_the_margin_is_not_part_of_the_calibrated_configuration(monkeypatch):
    # The cutoff stays valid: the margin reorders near ties and never changes a score, the candidate text the
    # reranker is sent, or which candidates pass the cutoff.
    reranker = Reranker({})
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK_TIE_MARGIN", raising=False)
    before = calibration_fingerprint(provider_calibration_metadata(reranker))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIE_MARGIN", ".2")
    assert calibration_fingerprint(provider_calibration_metadata(reranker)) == before
    assert CANDIDATE_FORMAT == "grounded-quotation-bundles-v2-corrections-first-rerank-cut-1500-2000"
