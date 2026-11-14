"""The prior records offered to the extractor come from both the keyword lane and, with vectors, a vector lane.

The keyword lane restricts the priors to the claims of the turns a keyword search of the new message finds. A
paraphrased change ("We relocated to Munich") shares no word with the record it changes ("My office is in
Berlin"), so for a person with many claims the keyword lane never offers it. The vector lane runs the existing
source vector search with the new message and offers the live claims of the turns it finds; the two lanes are
interleaved one for one, keyword first, under the same cap of 16. Without vectors nothing changes.
"""
from contextlib import closing
import json

import pytest

from colonymind.beliefs.source_claims import norm_value
from colonymind.beliefs.source_projection import (SIMILAR_TURNS, SourceClaimProjection, interleave_lanes,
                                                 one_witness_per_value)
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.source_vectors import SourceVectors
from colonymind.vector.indexes import EmbeddingIdentity, IndexCatalog
from colonymind.vector.store import VectorStore
from test_source_claim_projection import Model, claim
from test_source_episode_memory import episode

OFFICE = "My office is in Berlin."
MOVED = "We relocated to Munich last week, and I work there now."


class Pipeline:
    """Controlled embeddings: a text is one of three directions by the words it contains."""
    index_identity = EmbeddingIdentity("fixture", "fixture", "unknown", 3)

    def __init__(self, fail=False):
        self.fail = fail

    @staticmethod
    def vector(text):
        text = text.casefold()
        index = 0 if any(word in text for word in ("office", "relocated")) else (
            1 if any(word in text for word in ("kettle", "brew")) else 2)
        return [float(i == index) for i in range(3)]

    async def embed_batch(self, texts):
        return [self.vector(text) for text in texts]

    async def embed_query(self, text):
        if self.fail:
            raise RuntimeError("embedding endpoint unavailable")
        return self.vector(text)


class ChoosingModel(Model):
    """The scripted extractor links a change to the offered prior with the same predicate, as a model would."""

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        result = self.outputs.get(payload.get("message"), {})
        if kwargs.get("context", {}).get("task") == "source_claim_extraction" and result.get("choose_prior"):
            wanted = norm_value(result["predicate"].replace("_", " "))
            chosen = next((row["id"] for row in payload["prior_assertions"] if row["predicate"] == wanted), None)
            self.outputs[payload["message"]] = {k: v for k, v in result.items() if k != "choose_prior"} | {
                "prior_claim_id": chosen}
            try:
                return await super().complete(messages, **kwargs)
            finally:
                self.outputs[payload["message"]] = result
        return await super().complete(messages, **kwargs)


async def store(tmp_path, pipeline):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    vectors = VectorStore(str(tmp_path / "lancedb"), identity=pipeline.index_identity, catalog=IndexCatalog(ledger))
    await vectors.connect(3)
    await vectors.ensure_collections(3)
    return ledger, SourceVectors(ledger, vectors, pipeline)


async def drain(vectors):
    for _ in range(500):
        if not await vectors.process_one():
            return
    raise AssertionError("source vector jobs did not finish")


FILLERS = [f"My cousin number {index} studies in Munich." for index in range(20)]


async def person(ledger, projection, vectors, model, texts, *, contact="contact-a", start=0):
    for index, text in enumerate(texts, start):
        ledger.record_source(f"{contact}-t{index}", contact_id=contact, session_id=f"s{index}",
                             occurred_at=f"2026-03-01T{index // 60:02d}:{index % 60:02d}:00+00:00",
                             messages=[{"role": "user", "content": text}])
        assert await projection.process_one(model)
    await drain(vectors)


def outputs(extra=()):
    rows = {text: claim(text, "Munich", subject=f"My cousin number {index}", predicate=f"cousin_{index}_study_city")
            for index, text in enumerate(FILLERS)}
    rows[OFFICE] = claim(OFFICE, "Berlin")
    rows[MOVED] = claim(MOVED, "Munich", operation="change") | {"choose_prior": True}
    rows.update(extra)
    return rows


def claim_rows(ledger):
    with closing(ledger._connect()) as conn:
        return {row["id"]: dict(row) | json.loads(row["data_json"]) for row in conn.execute("SELECT * FROM source_claims")}


@pytest.mark.asyncio
async def test_with_vectors_a_paraphrased_change_is_offered_its_prior_and_supersedes_it(tmp_path):
    ledger, vectors = await store(tmp_path, Pipeline())
    projection = SourceClaimProjection(ledger, vectors=vectors)
    model = ChoosingModel(outputs())
    await person(ledger, projection, vectors, model, [OFFICE, *FILLERS])
    berlin, = [row for row in claim_rows(ledger).values() if row["value"] == "Berlin"]
    model.calls.clear()
    await person(ledger, projection, vectors, model, [MOVED], start=100)
    offered = [row["id"] for row in model.calls[0][0]["prior_assertions"]]
    assert berlin["id"] in offered and len(offered) <= 16
    rows = claim_rows(ledger)
    munich, = [row for row in rows.values() if row["value"] == "Munich" and row["subject_key"] == "speaker"]
    assert munich["operation"] == "change" and munich["prior_claim_id"] == berlin["id"]
    assert rows[berlin["id"]]["superseded_by"] == munich["id"]


@pytest.mark.asyncio
async def test_without_vectors_the_keyword_lane_alone_misses_the_paraphrased_prior(tmp_path):
    ledger, vectors = await store(tmp_path, Pipeline())
    projection = SourceClaimProjection(ledger)
    model = ChoosingModel(outputs())
    await person(ledger, projection, vectors, model, [OFFICE, *FILLERS])
    source = {"turn_id": "new", "contact_id": "contact-a", "session_id": "later"}
    offered = projection.prior(source, {"role": "user", "content": MOVED})
    assert "Berlin" not in {row["value"] for row in offered}
    assert offered == await projection.prior_claims(source, {"role": "user", "content": MOVED})


def prior_1_11(projection, source, message, limit=16):
    """``SourceClaimProjection.prior`` as released in 1.11.0, kept verbatim as the reference."""
    words = set(norm_value(message.get("content", "")).split())

    def relevance(row):
        basis = row.get('subject_basis', {}) if row.get('representation') == 'episode' else {}
        evidence = row['evidence'] + ' ' + basis.get('evidence', '')
        return len(words & set(norm_value(evidence).split())), row['recorded_at']

    hits = projection.ledger.search_sources(message.get("content", ""), contact_id=source["contact_id"],
                                            session_id=source["session_id"], limit=10)
    turn_ids = [hit["turn_id"] for hit in hits if hit["turn_id"] != source["turn_id"]]
    with closing(projection.ledger._connect()) as conn:
        rows = projection._rows(conn, source["contact_id"], source["session_id"], turn_ids=turn_ids or None)
        keys = list(dict.fromkeys((row['subject_key'], row['predicate'])
            for row in sorted(rows, key=relevance, reverse=True)
            if row.get('representation') == 'episode'))[:limit]
        for key in keys:
            rows.extend(projection._rows(conn, source['contact_id'], source['session_id'], key=key, limit=limit))
    rows = one_witness_per_value(row for row in rows if not row['superseded_by'] and not row['retracted_by'])
    rows.sort(key=relevance, reverse=True)
    return rows[:limit]


REPORT = "The harbour pump reset twice during a four-hour bench run; the cause was not established."


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [MOVED, "Anything about the kettle?", "harbour pump", "nothing matches zzzz"])
async def test_without_vectors_the_prior_set_is_identical_to_1_11(tmp_path, text):
    ledger, vectors = await store(tmp_path, Pipeline())
    projection = SourceClaimProjection(ledger)
    await person(ledger, projection, vectors, ChoosingModel(outputs({REPORT: episode(REPORT)})),
                 [OFFICE, REPORT, *FILLERS[:18]])
    source = {"turn_id": "new", "contact_id": "contact-a", "session_id": "later"}
    message = {"role": "user", "content": text}
    reference = prior_1_11(projection, source, message)
    assert reference
    assert projection.prior(source, message) == reference
    assert await projection.prior_claims(source, message) == reference
    # A vector search that fails leaves the keyword lane exactly as it was.
    failing = SourceClaimProjection(ledger, vectors=SourceVectors(ledger, vectors.store, Pipeline(fail=True)))
    assert await failing.prior_claims(source, message) == reference


def test_the_lanes_interleave_one_for_one_keyword_first_without_repeats_under_the_cap():
    def rows(prefix, count):
        return [{"id": f"{prefix}{i}", "subject_key": prefix, "predicate": "p", "value": f"{prefix}{i}"}
                for i in range(count)]
    keyword, meaning = rows("k", 10), rows("v", 10)
    merged = interleave_lanes(keyword, [keyword[1], *meaning], limit=16)
    assert [row["id"] for row in merged][:6] == ["k0", "k1", "k2", "v0", "k3", "v1"]
    assert len(merged) == 16 and len({row["id"] for row in merged}) == 16
    assert interleave_lanes(keyword, [], limit=16) == keyword
    # The same value from another row of the same key is offered once.
    same = [{"id": "x", "subject_key": "k", "predicate": "p", "value": "k0"}]
    assert [row["id"] for row in interleave_lanes(keyword[:1], same, limit=16)] == ["k0"]


class Crowding(Pipeline):
    """A session's checkpoint notes sit exactly on the query; the person's office record is near it, not on it."""

    @staticmethod
    def vector(text):
        text = text.casefold()
        if "checkpoint" in text or "relocated" in text:
            return [1.0, 0.0, 0.0]
        if "office" in text:
            return [0.6, 0.8, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_a_sessions_checkpoint_chunks_never_crowd_the_persons_claims_out_of_the_vector_lane(tmp_path):
    ledger, vectors = await store(tmp_path, Crowding())
    projection = SourceClaimProjection(ledger, vectors=vectors)
    model = ChoosingModel(outputs())
    await person(ledger, projection, vectors, model, [OFFICE, *FILLERS])
    # More checkpoint chunks of the current session than the lane reads, each closer to the query than the
    # office record. A checkpoint carries no claims; only the person's own turns can offer a prior.
    for index in range(SIMILAR_TURNS + 2):
        ledger.record_source(f"checkpoint-{index}", contact_id="contact-a", session_id="later", scope="session",
                             messages=[{"role": "user", "content": f"Checkpoint note {index} of this session."}])
    await drain(vectors)
    source = {"turn_id": "new", "contact_id": "contact-a", "session_id": "later"}
    message = {"role": "user", "content": MOVED}
    hits, _ = await vectors.search(MOVED, contact_id="contact-a", session_id="later", limit=10)
    assert {hit["scope"] for hit in hits} == {"session"}      # the combined search is all checkpoints
    assert "contact-a-t0" in await projection.similar_turns(source, message)
    assert "Berlin" in {row["value"] for row in await projection.prior_claims(source, message)}


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_erased_retracted_and_other_person_claims_are_never_offered_by_meaning(tmp_path):
    ledger, vectors = await store(tmp_path, Pipeline())
    projection = SourceClaimProjection(ledger, vectors=vectors)
    hamburg, fixed = ('My office is in Hamburg.', 'Correction: my office is in Bremen, not Hamburg.')
    annex = 'My office annex is in Bonn.'
    model = ChoosingModel(outputs({hamburg: claim(hamburg, 'Hamburg'), fixed: claim(fixed, 'Bremen', operation='correct') | {'choose_prior': True}, annex: claim(annex, 'Bonn', predicate='annex_location'), 'My office is in Paris.': claim('My office is in Paris.', 'Paris')}))
    await person(ledger, projection, vectors, model, [hamburg, fixed, annex, *FILLERS])
    await person(ledger, projection, vectors, model, ['My office is in Paris.'], contact='contact-b', start=50)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'contact-a', source_ids=['contact-a-t2'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    await drain(vectors)
    source = {'turn_id': 'new', 'contact_id': 'contact-a', 'session_id': 'later'}
    offered = {row['value'] for row in await projection.prior_claims(source, {'role': 'user', 'content': MOVED})}
    assert 'Bremen' in offered
    assert not offered & {'Hamburg', 'Bonn', 'Paris'}
