"""The extractor is shown the person's recent distinct claim keys, so a new assertion reuses one.

The list is names only (subject and predicate), newest first, at most twenty: never another person's, an erased
message's, or a retracted or superseded claim's, and never an episode's content-derived identity. It goes to the
extraction call only; the review judges the proposals against the source as before.
"""
import pytest

from colonymind.beliefs.source_claims import SYSTEM
from colonymind.beliefs.source_projection import KNOWN_KEYS, SourceClaimProjection
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_projection import Model, claim
from test_source_episode_memory import episode


def extraction_payloads(model):
    return [payload for payload, kwargs in model.calls
            if kwargs.get("context", {}).get("task") == "source_claim_extraction"]


def review_payloads(model):
    return [payload for payload, kwargs in model.calls
            if kwargs.get("context", {}).get("task") == "source_claim_review"]


async def say(ledger, projection, model, turn, text, *, contact="contact-a", at="2026-03-01T09:00:00+00:00"):
    ledger.record_source(turn, contact_id=contact, session_id="s-" + turn, occurred_at=at,
                         messages=[{"role": "user", "content": text}])
    assert await projection.process_one(model)


@pytest.mark.asyncio
async def test_the_key_list_is_bounded_and_a_first_message_has_none(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    texts = {f"My setting number {index} is v{index}.": index for index in range(KNOWN_KEYS + 5)}
    model = Model({text: claim(text, f"v{index}", predicate=f"setting_{index}") for text, index in texts.items()})
    for text, index in texts.items():
        await say(ledger, projection, model, f"t{index}", text, at=f"2026-03-01T09:{index:02d}:00+00:00")
    first, *_, last = extraction_payloads(model)
    assert "known_keys" not in first
    assert len(last["known_keys"]) == KNOWN_KEYS == 20
    assert last["known_keys"][0] == {"subject": "I", "predicate": f"setting {KNOWN_KEYS + 3}"}


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_the_extraction_prompt_lists_the_persons_recent_distinct_keys(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    projection = SourceClaimProjection(ledger)
    office, tea, moved = ('My office is in River.', 'My tea preference is green tea.', 'Correction: my office is in Lake.')
    other, gone = ('My office is in Harbour.', 'My locker code word is blue.')
    report = 'The pump reset twice during a four-hour bench run; the cause was not established.'
    new = 'My bike is at the station.'
    model = Model({office: claim(office, 'River'), tea: claim(tea, 'green tea', predicate='tea_preference'), moved: claim(moved, 'Lake', operation='correct', match_prior=True), other: claim(other, 'Harbour'), gone: claim(gone, 'blue', predicate='locker_word'), report: episode(report), new: claim(new, 'the station', predicate='bike_location')})
    await say(ledger, projection, model, 'office', office, at='2026-03-01T09:00:00+00:00')
    await say(ledger, projection, model, 'tea', tea, at='2026-03-02T09:00:00+00:00')
    await say(ledger, projection, model, 'moved', moved, at='2026-03-03T09:00:00+00:00')
    await say(ledger, projection, model, 'other', other, contact='contact-b', at='2026-03-04T09:00:00+00:00')
    await say(ledger, projection, model, 'gone', gone, at='2026-03-05T09:00:00+00:00')
    await say(ledger, projection, model, 'report', report, at='2026-03-06T09:00:00+00:00')
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'contact-a', source_ids=['gone'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    model.calls.clear()
    await say(ledger, projection, model, 'new', new, at='2026-03-07T09:00:00+00:00')
    payload, = extraction_payloads(model)
    assert payload['known_keys'] == [{'subject': 'I', 'predicate': 'office location'}, {'subject': 'I', 'predicate': 'tea preference'}]
    assert all(('known_keys' not in review for review in review_payloads(model)))
    assert 'known_keys' in SYSTEM
