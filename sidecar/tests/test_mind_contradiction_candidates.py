"""Contradiction candidates by meaning: the night pairs claims of one person under two different keys.

The contradictions stage paired two claims only when their ``(subject, predicate)`` key was identical, so the same
fact filed under two keys ("works at" and "employer") was never checked and the old value was recalled as
current. The night now also embeds each person's claim keys in one batch, pairs claims whose keys are close,
interleaves that list with the exact-key list round robin, and sends both to the same typed relation decision under
the same cap. Everything runs against the real ledger, claim projection and mind (``test_mind_consolidate``'s
fixture), with a stub embedder; nothing about keys may outlive the night.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import colonymind.mind.consolidate as consolidate
import colonymind.vector as vector
from colonymind.mind.consolidate import KEY_SIMILARITY_BOUND, RELATION_TASK, RELATIONS_PER_NIGHT, TASK_DIGEST, key_text
from test_mind_consolidate import CONTACT, OWNER, Fixture

WORKS_AT, EMPLOYER = key_text("speaker", "works at"), key_text("speaker", "employer")


class KeyEmbedder:
    """Embeds claim keys by topic: two key texts on one topic are the same vector (cosine 1), texts on different
    topics are orthogonal; a key text not in the table is a topic of its own. ``near`` gives a pair of texts an
    exact cosine instead. Every batch is recorded."""

    def __init__(self, topics=None, *, near=None, fail=None):
        self.topics = dict(topics or {})
        self.near = dict(near or {})            # text -> (other text, cosine)
        self.fail = fail
        self.batches = []
        self.axes = {}

    def _axis(self, name):
        return self.axes.setdefault(name, len(self.axes))

    def vector(self, text):
        if text in self.near:
            other, cosine = self.near[text]
            base = self.vector(other)
            extra = [0.0] * len(base)
            extra[self._axis(f"own:{text}") % len(base)] = 1.0
            return [cosine * x + math.sqrt(1 - cosine * cosine) * y for x, y in zip(base, extra)]
        values = [0.0] * 64
        values[self._axis(self.topics.get(text, f"own:{text}"))] = 1.0
        return values

    async def embed_batch(self, texts):
        self.batches.append(list(texts))
        if self.fail is not None:
            raise self.fail
        return [self.vector(text) for text in texts]


def relation_calls(fx):
    return [json.loads(messages[-1]["content"]) for messages, context in fx.router.calls
            if context["task"] == RELATION_TASK]


def retracted_by(fx):
    """What retracted each claim (None: still live). The mind's own settlement cites the night's note."""
    return {row["id"]: row["retracted_by"] for row in fx.claims()}


@pytest.fixture
def night(tmp_path, monkeypatch):
    """A fixture whose night has no digest work, with no sidecar embedding pipeline unless a test sets one."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setattr(vector, "_pipeline", None)
    fixture = Fixture(tmp_path)
    fixture.router.answers[TASK_DIGEST] = None
    yield fixture
    fixture.store.close()


# ---------------------------------------------------------------------------
# 1. The same fact under two keys becomes a candidate, is decided, and the mind keeps the newer statement
# ---------------------------------------------------------------------------

async def test_works_at_and_employer_under_two_keys_become_one_candidate_and_the_newer_is_kept(night):
    fx = night
    embedder = KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job"})
    fx.mind.consolidation.embedder = embedder
    first = await fx.fact("t-1", OWNER, "s-1", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(days=1)
    second = await fx.fact("t-2", OWNER, "s-2", "My employer is Globex.", "Globex", predicate="employer")

    result = await fx.mind.consolidate()
    assert [sorted(batch) for batch in embedder.batches] == [sorted([WORKS_AT, EMPLOYER])]   # one batch, its keys
    call, = relation_calls(fx)
    assert call["about"] == "your works at and employer"                  # the pair names both keys
    assert call["older"]["said"] == "I work at Acme." and call["newer"]["said"] == "My employer is Globex."
    assert result["counts"]["contradictions"] == 1 and result["counts"]["keys_embedded"] == 2

    kept = retracted_by(fx)
    assert kept[first].startswith("note:") and kept[second] is None       # settled by the mind, under two keys
    assert fx.messages("contradiction") == []                             # and the owner is asked nothing


async def test_a_cross_key_pair_decided_on_an_earlier_night_is_settled_on_a_night_without_embeddings(night):
    """The settlement rests on the recorded decision, not on the night's vectors: a pair an earlier night decided
    "replaces" is settled when the embedding endpoint is down, and is never judged again."""
    fx = night
    first = await fx.fact("t-1", OWNER, "s-1", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(hours=2)
    second = await fx.fact("t-2", OWNER, "s-2", "My employer is Globex.", "Globex", predicate="employer")
    fx.mind.consolidation._record_relation({"id": first}, {"id": second}, "replaces", contact_id=OWNER,
                                           by="note:earlier", now=fx.now)        # as an earlier night left it

    fx.mind.consolidation.embedder = KeyEmbedder(fail=ConnectionError("embedding endpoint down"))
    fx.shift(days=1)
    result = await fx.mind.consolidate()
    assert relation_calls(fx) == [] and result["counts"]["contradictions"] == 1
    kept = retracted_by(fx)
    assert kept[first].startswith("note:") and kept[second] is None


async def test_keeping_the_newer_drops_every_live_copy_of_the_older_value_under_its_own_key(night):
    """A cross-key pair's two claims are filed under two keys: keeping the newer retracts the older value,
    including a copy of it said before under the first key (the copies are read under the dropped claim's key)."""
    fx = night
    fx.mind.consolidation.embedder = KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job"})
    copy = await fx.fact("t-1", OWNER, "s-1", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(hours=1)
    witness = await fx.fact("t-2", OWNER, "s-2", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(hours=1)
    newest = await fx.fact("t-3", OWNER, "s-3", "My employer is Globex.", "Globex", predicate="employer")
    result = await fx.mind.consolidate()
    assert copy != witness and result["counts"]["contradictions"] == 1    # one witness per value: the newest
    kept = retracted_by(fx)
    assert kept[witness] == kept[copy] and kept[copy].startswith("note:")
    assert kept[newest] is None


# ---------------------------------------------------------------------------
# 2. Identical keys are still paired
# ---------------------------------------------------------------------------

async def test_identical_keys_are_still_paired_with_the_embedder_on(night):
    fx = night
    fx.mind.consolidation.embedder = KeyEmbedder()                        # every key its own topic
    first = await fx.fact("t-1", OWNER, "s-1", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    await fx.fact("t-2", OWNER, "s-2", "I drive a Civic.", "Civic", predicate="car")
    fx.shift(hours=1)
    second = await fx.fact("t-3", OWNER, "s-3", "My office is room 7.", "room 7")

    result = await fx.mind.consolidate()
    assert [call["about"] for call in relation_calls(fx)] == ["your office location"]
    assert result["counts"]["contradictions"] == 1
    kept = retracted_by(fx)
    assert kept[first].startswith("note:") and kept[second] is None      # the exact key's pair is settled as before


# ---------------------------------------------------------------------------
# 3. Different contacts are never paired
# ---------------------------------------------------------------------------

async def test_claims_of_two_people_are_never_paired_however_close_their_keys(night):
    fx = night
    home, town = key_text("speaker", "home city"), key_text("speaker", "home town")
    embedder = KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job", home: "home", town: "home"})
    fx.mind.consolidation.embedder = embedder
    await fx.fact("o-1", OWNER, "s-o1", "I work at Acme.", "Acme", predicate="works_at")
    await fx.fact("o-2", OWNER, "s-o2", "My home city is Lisbon.", "Lisbon", predicate="home_city")
    fx.shift(hours=1)
    await fx.fact("c-1", CONTACT, "s-c1", "My employer is Globex.", "Globex", predicate="employer")
    await fx.fact("c-2", CONTACT, "s-c2", "My home town is Porto.", "Porto", predicate="home_town")

    await fx.mind.consolidate()
    assert sorted(sorted(batch) for batch in embedder.batches) == [sorted([EMPLOYER, town]),
                                                                   sorted([WORKS_AT, home])]  # one per person
    assert relation_calls(fx) == [] and set(retracted_by(fx).values()) == {None}


# ---------------------------------------------------------------------------
# 4. Claims without overlapping validity are not paired; a pair below the bound is not a candidate
# ---------------------------------------------------------------------------

def claim_row(ident, predicate, value, *, valid_from=None, valid_to=None, recorded="2026-09-01T00:00:00+00:00"):
    return {"id": ident, "subject_key": "speaker", "subject": "I", "predicate": predicate, "value": value,
            "valid_from": valid_from, "valid_to": valid_to, "recorded_at": recorded, "representation": "assertion"}


def test_claims_whose_validity_does_not_overlap_are_not_paired(night):
    consolidation = night.mind.consolidation
    vectors = {("speaker", "works at"): [1.0, 0.0], ("speaker", "employer"): [1.0, 0.0]}
    before = claim_row("claim:a", "works at", "Acme", valid_from="2019-01-01T00:00:00+00:00",
                       valid_to="2021-01-01T00:00:00+00:00")
    after = claim_row("claim:b", "employer", "Globex", valid_from="2021-01-01T00:00:00+00:00")
    during = claim_row("claim:c", "employer", "Initech", valid_from="2020-06-01T00:00:00+00:00",
                       recorded="2026-09-02T00:00:00+00:00")
    assert consolidation._similar([before, after], vectors) == []
    pairs = consolidation._similar([before, after, during], vectors)
    assert [(a["id"], b["id"]) for a, b, _ in pairs] == [("claim:a", "claim:c")]


def test_a_pair_of_keys_below_the_bound_is_not_a_candidate_and_one_just_above_it_is(night):
    consolidation = night.mind.consolidation
    a = claim_row("claim:a", "works at", "Acme")
    b = claim_row("claim:b", "employer", "Globex", recorded="2026-09-02T00:00:00+00:00")
    same = claim_row("claim:c", "employer", "acme", recorded="2026-09-03T00:00:00+00:00")   # the same value
    below = math.sqrt(1 - (KEY_SIMILARITY_BOUND - 0.01) ** 2)
    far = {("speaker", "works at"): [1.0, 0.0], ("speaker", "employer"): [KEY_SIMILARITY_BOUND - 0.01, below]}
    just = KEY_SIMILARITY_BOUND + 1e-9
    near = {("speaker", "works at"): [1.0, 0.0], ("speaker", "employer"): [just, math.sqrt(1 - just ** 2)]}
    assert consolidation._similar([a, b], far) == []
    assert [(x["id"], y["id"]) for x, y, _ in consolidation._similar([a, b, same], near)] == [("claim:a", "claim:b")]


@pytest.mark.parametrize("measured, pairs, status", [(0.8, 1, None), (0.9, 0, None), ("below", 0, "key_bound_below_floor"),
                                                     (None, 0, "key_bound_unmeasured")])
async def test_the_night_uses_the_bound_measured_for_its_embedding(night, monkeypatch, tmp_path, measured, pairs,
                                                                   status):
    """Two keys 0.85 alike are a pair under a bound of 0.8 measured for the night's embedding and not under 0.9; an
    embedding whose measurement kept no bound above the precision floor embeds nothing; one never measured runs on
    the default bound (stricter than 0.85) and the night says so."""
    import dataclasses

    from colonymind.mind import contradiction_keys
    from colonymind.vector.indexes import EmbeddingIdentity
    fx = night
    ident = EmbeddingIdentity(requested_model="embed-model", served_model="served-embed", declared_revision="unknown",
                              dimensions=64)
    entries = [] if measured is None else [{
        "dev_set": {"sha256": contradiction_keys.DEV_SET_SHA256}, "precision_floor": 0.8, "at_bound": None,
        "embedding": {**dataclasses.asdict(ident), "fingerprint": ident.fingerprint}, "pairs": [],
        "bound": 0.5 if measured == "below" else measured, "floor_met": measured != "below"}]
    (tmp_path / "bounds.json").write_text(json.dumps({"measurements": entries}))
    monkeypatch.setattr(contradiction_keys, "BOUNDS", tmp_path / "bounds.json")
    embedder = KeyEmbedder(near={EMPLOYER: (WORKS_AT, 0.85)})
    embedder.index_identity = ident
    fx.mind.consolidation.embedder = embedder
    await fx.fact("t-1", OWNER, "s-1", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(days=1)
    await fx.fact("t-2", OWNER, "s-2", "My employer is Globex.", "Globex", predicate="employer")

    result = await fx.mind.consolidate()
    assert len(relation_calls(fx)) == pairs
    assert len(embedder.batches) == (0 if measured == "below" else 1)
    flags = {key for key in ("key_bound_below_floor", "key_bound_unmeasured") if key in result["counts"]}
    assert flags == ({status} if status else set())
    assert KEY_SIMILARITY_BOUND > 0.85


# ---------------------------------------------------------------------------
# 5. The cap of 8 holds: the exact-key and the vector lists interleaved round robin
# ---------------------------------------------------------------------------

EXACT = ("office location", "desk number", "parking spot", "locker number", "badge colour", "shift start")
CROSS = (("works at", "employer"), ("home city", "lives in"), ("phone number", "mobile number"),
         ("car", "vehicle"), ("gym", "fitness club"), ("team", "department"))


async def test_the_cap_of_eight_holds_with_the_two_lists_interleaved_round_robin(night):
    fx = night
    fx.router.answers[RELATION_TASK] = {"relation": "adds"}               # decided, never asked
    topics = {}
    for index, (left, right) in enumerate(CROSS):
        topics[key_text("speaker", left)] = topics[key_text("speaker", right)] = f"topic-{index}"
    fx.mind.consolidation.embedder = KeyEmbedder(topics)
    for index, predicate in enumerate(EXACT):
        await fx.fact(f"e-{index}-a", OWNER, "s", f"My {predicate} is A{index}.", f"A{index}", predicate=predicate)
        fx.shift(minutes=1)
        await fx.fact(f"e-{index}-b", OWNER, "s", f"My {predicate} is B{index}.", f"B{index}", predicate=predicate)
        fx.shift(minutes=1)
    for index, (left, right) in enumerate(CROSS):
        await fx.fact(f"x-{index}-a", OWNER, "s", f"My {left} is C{index}.", f"C{index}", predicate=left)
        fx.shift(minutes=1)
        await fx.fact(f"x-{index}-b", OWNER, "s", f"My {right} is D{index}.", f"D{index}", predicate=right)
        fx.shift(minutes=1)

    first = await fx.mind.consolidate()
    abouts = [call["about"] for call in relation_calls(fx)]
    assert len(abouts) == RELATIONS_PER_NIGHT == 8 and first["counts"]["compatible"] == 8
    assert [" and " in about for about in abouts] == [False, True] * 4   # exact, similar, exact, similar, ...
    fx.shift(days=1)
    await fx.mind.consolidate()
    rest = [call["about"] for call in relation_calls(fx)][8:]
    assert len(rest) == 4 and [" and " in about for about in rest] == [False, True] * 2
    assert len(set(abouts + rest)) == 12                                  # every pair decided once


async def test_a_slow_embedder_never_starves_the_rest_of_the_night(night, monkeypatch):
    """Every person's batch is answered within its own deadline, yet all of them together would outlast the whole
    night. The pass is bounded: the exact-key decision, the digests and the episodes still run, the persons with
    the newest statements are embedded first and their pairs are still decided, and the rest wait for a later
    night (scaled: a 4 second night, half a second per person, 0.3 seconds per batch, 16 persons)."""
    fx = night
    monkeypatch.setattr(consolidate, "RUN_DEADLINE_S", 4.0)
    monkeypatch.setattr(consolidate, "KEY_EMBED_DEADLINE_S", 0.5)

    class Slow(KeyEmbedder):
        async def embed_batch(self, texts):
            await asyncio.sleep(0.3)
            return await super().embed_batch(texts)

    people = [f"p-{10 + index}" for index in range(16)]
    topics = {}
    for index in range(len(people)):
        topics[key_text("speaker", f"job {index}")] = topics[key_text("speaker", f"employer {index}")] = f"t{index}"
    embedder = Slow(topics)
    fx.mind.consolidation.embedder = embedder
    await fx.fact("o-1", OWNER, "s-o", "My office is room 4.", "room 4")
    fx.shift(minutes=1)
    await fx.fact("o-2", OWNER, "s-o", "My office is room 7.", "room 7")
    for index, person in enumerate(people):
        fx.shift(minutes=1)
        await fx.fact(f"{person}-a", person, f"s-{person}", f"My job {index} is Acme.", "Acme",
                      predicate=f"job_{index}")
        fx.shift(minutes=1)
        await fx.fact(f"{person}-b", person, f"s-{person}", f"My employer {index} is Globex.", "Globex",
                      predicate=f"employer_{index}")

    result = await fx.mind.consolidate()
    assert "deadline" not in result["errors"]
    assert {"contradictions", "digests", "episodes"} <= set(result["done"])
    abouts = [call["about"] for call in relation_calls(fx)]
    assert "your office location" in abouts                                  # the exact key is decided
    assert any(" and " in about for about in abouts)                         # pairs by meaning still found
    newest = len(people) - 1
    assert sorted(embedder.batches[0]) == sorted([key_text("speaker", f"job {newest}"),
                                                  key_text("speaker", f"employer {newest}")])
    assert 0 < len(embedder.batches) < len(people)
    assert result["counts"]["key_embedding_deferred"] >= len(people) - len(embedder.batches)
    assert "key_embedding_failed" not in result["counts"]


# ---------------------------------------------------------------------------
# 6. Without embeddings, behaviour is identical to 1.11.0
# ---------------------------------------------------------------------------

async def plant(fx):
    first = await fx.fact("t-1", OWNER, "s-1", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    await fx.fact("t-2", OWNER, "s-2", "I work at Acme.", "Acme", predicate="works_at")
    fx.shift(hours=1)
    await fx.fact("t-3", OWNER, "s-3", "My employer is Globex.", "Globex", predicate="employer")
    fx.shift(hours=1)
    second = await fx.fact("t-4", OWNER, "s-4", "My office is room 7.", "room 7")
    return first, second


@pytest.mark.parametrize("embedder", [None, KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job"},
                                                        fail=TimeoutError("no answer")),
                                      "wrong-length", "ragged"])
async def test_without_embeddings_the_night_pairs_exact_keys_only(night, embedder):
    fx = night
    if embedder == "wrong-length":
        class Short(KeyEmbedder):
            async def embed_batch(self, texts):
                self.batches.append(list(texts))
                return [self.vector(texts[0])]
        embedder = Short({WORKS_AT: "job", EMPLOYER: "job"})
    elif embedder == "ragged":
        class Ragged(KeyEmbedder):
            async def embed_batch(self, texts):
                self.batches.append(list(texts))
                return [self.vector(text)[: 64 - index] for index, text in enumerate(texts)]
        embedder = Ragged({WORKS_AT: "job", EMPLOYER: "job"})
    fx.mind.consolidation.embedder = embedder
    first, _ = await plant(fx)
    result = await fx.mind.consolidate()
    assert [call["about"] for call in relation_calls(fx)] == ["your office location"]
    kept = retracted_by(fx)
    assert kept.pop(first).startswith("note:") and set(kept.values()) == {None}   # the office pair only
    assert "keys_embedded" not in result["counts"]
    if embedder is not None:
        assert result["counts"]["key_embedding_failed"] == 1


async def test_the_night_reads_the_sidecars_embedding_pipeline_when_none_is_given(night, monkeypatch):
    fx = night
    pipeline = KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job"})
    monkeypatch.setattr(vector, "_pipeline", pipeline)
    assert fx.mind.consolidation.embedder is None
    await plant(fx)
    await fx.mind.consolidate()
    assert len(pipeline.batches) == 1
    assert sorted(call["about"] for call in relation_calls(fx)) == ["your office location",
                                                                    "your works at and employer"]


# ---------------------------------------------------------------------------
# 7. Nothing about keys is written to any store
# ---------------------------------------------------------------------------

def table_counts(root: Path):
    """Rows per table in every SQLite database under ``root``, keyed by the file's path relative to it."""
    counts = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.read_bytes()[:16] != b"SQLite format 3\x00":
            continue
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                counts[(str(path.relative_to(root)), table)] = conn.execute(
                    f'SELECT count(*) FROM "{table}"').fetchone()[0]
    return counts


async def test_no_key_text_or_vector_is_written_to_any_store(tmp_path, monkeypatch):
    """The same store and night with and without the embedder, the relation decision leaving every pair undecided:
    the embedder ran, yet every database holds the same rows and the state directories the same files."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setattr(vector, "_pipeline", None)
    runs = {}
    for name, embedder in (("without", None), ("with", KeyEmbedder({WORKS_AT: "job", EMPLOYER: "job"}))):
        fx = Fixture(tmp_path / name)
        fx.router.answers[TASK_DIGEST] = None
        fx.router.answers[RELATION_TASK] = {"relation": "unsure"}
        fx.mind.consolidation.embedder = embedder
        await plant(fx)
        result = await fx.mind.consolidate()
        fx.mind.concerns.close()
        fx.store.close()
        runs[name] = (table_counts(tmp_path / name), sorted(str(path.relative_to(tmp_path / name))
                                                            for path in (tmp_path / name).rglob("*")), result)
    assert runs["with"][2]["counts"]["keys_embedded"] == 3 and runs["with"][2]["counts"]["undecided"] == 2
    assert runs["without"][2]["counts"]["undecided"] == 1
    assert runs["with"][0] == runs["without"][0] and runs["with"][0]
    assert runs["with"][1] == runs["without"][1]
