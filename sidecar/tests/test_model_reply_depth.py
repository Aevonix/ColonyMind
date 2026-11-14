"""Excessively nested model output follows the parser's unusable-reply path."""
from types import SimpleNamespace

import pytest

from colonymind.beliefs import source_claims
from colonymind.commitments import extract
from colonymind.instinct import deliberate as instinct_deliberate
from colonymind.mind import consolidate, deliberate, drift, ideas, lessons, opinions, plain


@pytest.fixture(params=["", "The answer is: ", "```json\n"], ids=["bare", "prose", "fenced"])
def deep_reply(request):
    # The closing brace reaches parsers that also try an object inside prose.
    # The object is incomplete on every interpreter, regardless of its depth limit.
    return request.param + '{"k":' * 8000 + '}' + ('\n```' if request.param == '```json\n' else '')


def test_ideas_rejects_deep_reply(deep_reply):
    with pytest.raises(ideas.AnswerUnavailable):
        ideas.answer(deep_reply, "offers")


def test_consolidation_rejects_deep_reply(deep_reply):
    assert consolidate._parse_object(deep_reply) is None


def test_deliberation_rejects_deep_reply(deep_reply):
    assert deliberate.parse_proposal(deep_reply) is None


@pytest.mark.asyncio
async def test_drift_rejects_deep_reply_and_keeps_usage(deep_reply):
    response = SimpleNamespace(content=deep_reply, usage={"total_tokens": 17})

    async def complete(**kwargs):
        return response

    check = drift.DriftCheck(store=None, ledger=None, mind_state=None, owner_id=None, clock=lambda: None,
                             router=SimpleNamespace(supports_function_routing=True, complete=complete))
    assert await check._ask(drift.Record()) == ({}, response, 17, "unparsable")


def test_claim_review_rejects_deep_reply(deep_reply):
    with pytest.raises(source_claims.SourceClaimOutputError, match="invalid_claim_review_json"):
        source_claims.validated_review(deep_reply, 1)


def test_claim_extraction_rejects_deep_reply(deep_reply):
    diagnostics = {}
    with pytest.raises(source_claims.SourceClaimOutputError, match="invalid_claim_array_json"):
        source_claims.validated_claims(deep_reply, message="hello", prior=[], observed_at=None,
                                       diagnostics=diagnostics)
    assert diagnostics["invalid_array_count"] == 1


def test_plain_summary_rejects_deep_reply(deep_reply):
    assert plain._object(deep_reply) is None


def test_instinct_decision_rejects_deep_reply(deep_reply):
    assert instinct_deliberate._object(deep_reply) is None


def test_opinion_rejects_deep_reply(deep_reply):
    with pytest.raises(opinions.OpinionOutputError, match="invalid_json"):
        opinions.validate(deep_reply, None)


def test_commitments_rejects_deep_reply():
    with pytest.raises(ValueError, match="unparsable_output"):
        extract.parse_items('[' * 8000 + ']')


def test_lesson_reflection_rejects_deep_reply():
    assert lessons.Lessons._reflection_ops('{"lesson_ops":' + '[' * 8000 + ']}', None) is None
