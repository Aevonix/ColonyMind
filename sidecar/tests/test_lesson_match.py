"""Lesson selection by meaning (``P/mind/lesson_match.py``, ``P/mind/lessons.py``).

Which lesson reaches a task body or an owner turn is the cosine of the lesson's vector (its title and when-to-use, the
index's document format) and the work's (the index's query format), at or above the bound measured for the running
embedding on a dev set pinned by hash. These tests pin the dev set, the bound's selection and lookup, the lesson
vectors (a derived cache that follows the served lessons), the owner turn's latency rules (the recall lane's vector
reused, one embed call inside the turn's budget otherwise, nothing by meaning on a timeout) and the diagnostics.
Every embedder here is a stub.
"""

from __future__ import annotations

from lesson_support import vectors_settled

import asyncio
import dataclasses
import hashlib
import inspect
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import colonymind.vector as vector_module
from colonymind.memory import turn_budget
from colonymind.memory.turn_budget import TurnBudget
from colonymind.mind import lesson_match as match
from colonymind.mind import lessons as lessons_module
from colonymind.turns import source_vectors
from colonymind.vector.indexes import EmbeddingIdentity
from test_mind_lessons import WIDTH, ConceptEmbedder, concept_vector, embedder, identity  # noqa: F401  (fixture)


# -- the dev set --------------------------------------------------------------------------------

def test_the_dev_set_is_pinned_by_hash_and_a_changed_one_is_refused(tmp_path):
    raw = match.DEV_SET.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == match.DEV_SET_SHA256
    changed = tmp_path / "dev.json"
    changed.write_bytes(raw.replace(b"Order codes by channel", b"Order codes by the channel"))
    with pytest.raises(ValueError, match="sha256"):
        match.load_dev_set(changed)
    assert len(match.load_dev_set(changed, sha256=None)) == 80


def test_the_dev_set_is_forty_related_and_forty_unrelated_pairs_half_of_those_declared_near_misses():
    pairs = match.load_dev_set()
    assert len(pairs) == 80 and len({pair["id"] for pair in pairs}) == 80
    assert sum(pair["related"] for pair in pairs) == 40
    unrelated = [pair for pair in pairs if not pair["related"]]
    assert all(isinstance(pair.get("near_miss"), bool) for pair in unrelated)
    assert sum(pair["near_miss"] for pair in unrelated) == 20
    assert not any("near_miss" in pair for pair in pairs if pair["related"])
    for pair in pairs:
        assert pair["work"]["kind"] in {"turn", "task"} and pair["lesson"]["title"] and pair["lesson"]["when_to_use"]


def test_near_misses_share_terms_and_related_pairs_few():
    """A near miss shares at least two of the lesson's terms but not its situation; a related pair is the situation
    worded differently, sharing at most three terms (few; my assumption)."""
    for pair in match.load_dev_set():
        document, work = match.pair_texts(pair)
        lesson = lessons_module.terms(f"{pair['lesson']['title']} {pair['lesson']['when_to_use']}")
        shared = len(lesson & lessons_module.terms(work))
        if pair.get("near_miss"):
            assert shared >= 2, pair["id"]
        if pair["related"]:
            assert shared <= 3, pair["id"]


def test_the_dev_set_lists_its_sources_none_held_out_and_uses_generic_words():
    data = json.loads(match.DEV_SET.read_text())
    repository = Path(__file__).resolve().parents[2]
    assert data["sources"] and all(pair["source"] for pair in data["pairs"])
    for source in {pair["source"] for pair in data["pairs"]} | set(data["sources"]):
        assert not re.search(r"held.?out", source, re.I), source
        assert (repository / source.split(" ")[0]).exists(), source
    text = match.DEV_SET.read_text()
    assert "\u2014" not in text and not re.search(r"held.?out split", text.replace("never a held-out split", ""))
    assert not re.search(r"\b\d{3}-\d{3}-\d{4}\b", text)


def test_the_texts_are_the_ones_the_runtime_embeds():
    [pair] = [pair for pair in match.load_dev_set() if pair["work"]["kind"] == "task"][:1]
    document, work = match.pair_texts(pair)
    assert document == match.lesson_text(pair["lesson"]["title"], pair["lesson"]["when_to_use"])
    assert work == match.task_text(pair["work"]["title"], pair["work"]["topic"], pair["work"]["concern"])
    assert match.lesson_text(" Order  codes ", "an order code is asked for") == (
        "Order codes. When an order code is asked for")
    assert match.task_text("Research order codes", "", "research order codes") == (
        "Research order codes research order codes")


async def test_the_dev_set_embeds_lessons_as_documents_in_one_batch_and_works_as_queries():
    pipeline = ConceptEmbedder()
    pairs = match.load_dev_set()
    scored = await match.score_dev_set(pipeline, pairs)
    assert len(pipeline.batches) == 1 and len(pipeline.batches[0]) == 20
    assert set(pipeline.batches[0]) == {match.pair_texts(pair)[0] for pair in pairs}
    assert set(pipeline.queries) == {match.pair_texts(pair)[1] for pair in pairs}
    assert [row["id"] for row in scored] == [pair["id"] for pair in pairs]
    assert all(-1.0 <= row["score"] <= 1.0 for row in scored)


# -- the bound ------------------------------------------------------------------------------------

SYNTHETIC = ([{"id": f"r{n}", "related": True, "score": score} for n, score in enumerate((0.9, 0.8, 0.7, 0.4))]
             + [{"id": f"u{n}", "related": False, "score": score} for n, score in enumerate((0.75, 0.3, 0.2, 0.1))])


def test_select_bound_reproduces_a_known_choice():
    """At 0.4 every related pair is found and 4 of the 5 found are related (0.8, the floor): the bound sits halfway
    to the next lower score, 0.3."""
    chosen = match.select_bound(SYNTHETIC)
    assert chosen["floor_met"] is True and chosen["bound"] == pytest.approx(0.35)
    assert (chosen["at_bound"]["tp"], chosen["at_bound"]["fp"]) == (4, 1)
    assert chosen["at_bound"]["precision"] == pytest.approx(0.8) and chosen["at_bound"]["recall"] == 1.0
    tighter = match.select_bound(SYNTHETIC, precision_floor=0.9)
    assert tighter["bound"] == pytest.approx(0.775) and tighter["at_bound"]["tp"] == 2


def shipped(model):
    data = json.loads(match.BOUNDS.read_text())
    [entry] = [entry for entry in data["measurements"] if entry["embedding"]["requested_model"] == model]
    return entry


def test_every_shipped_bound_recomputes_from_its_scores_and_names_only_public_models():
    data = json.loads(match.BOUNDS.read_text())
    assert {entry["embedding"]["requested_model"] for entry in data["measurements"]} == {
        "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B"}
    for entry in data["measurements"]:
        assert entry["dev_set"]["sha256"] == match.DEV_SET_SHA256 and entry["precision_floor"] == match.PRECISION_FLOOR
        again = match.select_bound(entry["pairs"], precision_floor=entry["precision_floor"])
        assert (again["bound"], again["floor_met"]) == (entry["bound"], entry["floor_met"])
        assert re.fullmatch(r"[0-9a-f]{40}", entry["embedding"]["declared_revision"])
    text = match.BOUNDS.read_text()
    assert not re.search(r"https?:|localhost|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|/home/|/Users/", text)


def test_the_default_bound_is_no_looser_than_any_shipped_bound():
    bounds = [entry["bound"] for entry in json.loads(match.BOUNDS.read_text())["measurements"] if entry["floor_met"]]
    assert bounds and match.DEFAULT_BOUND >= max(bounds)


def as_mapping(ident):
    return {**dataclasses.asdict(ident), "fingerprint": ident.fingerprint}


def test_the_bound_is_the_one_measured_for_the_embedding_and_below_the_floor_is_nothing(monkeypatch):
    monkeypatch.delenv(match.DEPLOYMENT_BOUNDS, raising=False)
    large, small = shipped("Qwen/Qwen3-Embedding-8B"), shipped("Qwen/Qwen3-Embedding-0.6B")
    assert large["floor_met"] and not small["floor_met"]
    assert match.bound_for(large["embedding"]) == (large["bound"], "measured")
    assert match.bound_for(small["embedding"]) == (None, "below_floor")
    # Another width, another query instruction, another model, or no identity: not measured, the default.
    for changed in ({"dimensions": 1024}, {"query_format": "prefix-v1:other"}, {"requested_model": "other/model"}):
        assert match.bound_for({**large["embedding"], **changed}) == (match.DEFAULT_BOUND, "unmeasured")
    assert match.bound_for(None) == (match.DEFAULT_BOUND, "unmeasured")
    assert match.bound_for(as_mapping(identity())) == (match.DEFAULT_BOUND, "unmeasured")


def test_a_deployments_bounds_file_overrides_only_the_embeddings_it_measured(tmp_path, monkeypatch):
    large, small = shipped("Qwen/Qwen3-Embedding-8B"), shipped("Qwen/Qwen3-Embedding-0.6B")
    own = tmp_path / "bounds.json"
    match.record({**small, "precision_floor": 0.6, "bound": 0.3, "floor_met": True, "at_bound": None}, own)
    monkeypatch.setenv(match.DEPLOYMENT_BOUNDS, str(own))
    assert match.bound_for(small["embedding"]) == (0.3, "measured")
    assert match.bound_for(large["embedding"]) == (large["bound"], "measured")
    monkeypatch.setenv(match.DEPLOYMENT_BOUNDS, str(tmp_path / "missing.json"))
    assert match.bound_for(small["embedding"]) == (None, "below_floor")


def test_the_command_line_measures_and_records_without_a_host_in_the_record(tmp_path, monkeypatch, capsys):
    async def made(*args, **kwargs):
        return ConceptEmbedder("measured/model")
    monkeypatch.setattr("colonymind.mind.contradiction_keys._sidecar_embedder", made)
    out, bounds = tmp_path / "report.json", tmp_path / "bounds.json"
    assert match.main(["--base-url", "http://127.0.0.1:9/v1", "--model", "measured/model", "--out", str(out),
                       "--record", str(bounds)]) == 0
    report = json.loads(out.read_text())
    assert report["dev_set"] == {"name": "lesson-match-dev-1", "sha256": match.DEV_SET_SHA256, "pairs": 80,
                                 "related": 40}
    [entry] = json.loads(bounds.read_text())["measurements"]
    assert entry["embedding"]["requested_model"] == "measured/model" and len(entry["pairs"]) == 80
    assert "127.0.0.1" not in bounds.read_text() and "127.0.0.1" not in capsys.readouterr().out


# -- selection (the mind's fixture) ----------------------------------------------------------------

from test_mind_lessons import GUEST, admit, candidate, fields, fx, sections, served  # noqa: E402,F401
from test_mind_loop import pinned_clock  # noqa: E402,F401  (autouse: a Fixture needs it)
from test_turn_source_evidence import source_app  # noqa: E402,F401  (pytest fixture)

PARAPHRASE = "Purchase 5120 arrived over email: which reference should it carry?"


def crate_lesson(fx):
    return admit(fx, fields=fields(signature="topic:crate-labels", title="Crate labels list the batch",
                                   when_to_use="crate labels are printed for a batch",
                                   content="Put the batch number first on every crate label."), lineage=())


async def test_a_paraphrased_task_gets_the_lesson_by_meaning_and_a_near_miss_does_not(fx, embedder):
    lessons = fx.mind.lessons
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    paraphrase = candidate(1, topic="purchase references")
    paraphrase.title, paraphrase.concern = "Label purchase 7731 from the chat", "which identifier it needs"
    assert not lessons_module.terms(f"{lesson.title} {lesson.when_to_use}") & lessons_module.terms(
        match.task_text(paraphrase.title, paraphrase.topic, paraphrase.concern))
    lines, ids, kinds = await lessons.task_lessons(paraphrase)
    assert ids == [lesson.id] and kinds == {lesson.id: "meaning"} and lines == [lesson.line()]
    # A near miss shares the lesson's words (order, code, channel) but names other things: under the bound.
    near = candidate(2, topic="dashboard channels")
    near.title, near.concern = "Order the channels on the dashboard alphabetically", "then the code review"
    assert match.cosine(concept_vector(match.task_text(near.title, near.topic, near.concern)),
                        concept_vector(match.lesson_text(lesson.title, lesson.when_to_use))) < match.DEFAULT_BOUND
    assert (await lessons.task_lessons(near))[1] == []


async def test_own_class_lessons_come_first_and_the_cap_holds(fx, embedder):
    lessons = fx.mind.lessons
    own = admit(fx, fields=fields(), lineage=())
    trial = admit(fx, fields=fields(kind="pitfall", content="Never use Z as the channel letter."), status="candidate",
                  verified="none", origin="reflector", lineage=())
    related = admit(fx, fields=fields(signature="topic:order-codes-2", title="Purchase references by channel",
                                      when_to_use="a purchase reference is needed",
                                      content="The channel letter comes first."), lineage=())
    await vectors_settled(lessons)
    lines, ids, kinds = await lessons.task_lessons(candidate(1))
    assert ids == [own.id, trial.id] and kinds == {own.id: "own_class", trial.id: "own_class"}
    # Another class: the related active lesson by meaning; never a candidate outside its class.
    other = candidate(3, topic="purchase references")
    lines, ids, kinds = await lessons.task_lessons(other)
    assert related.id in ids and trial.id not in ids and len(ids) <= lessons_module.TASK_LESSONS
    assert all(kinds[ident] == "meaning" for ident in ids)


async def test_related_lessons_come_by_descending_cosine(fx, embedder):
    lessons = fx.mind.lessons
    close = admit(fx, fields=fields(signature="topic:a", title="Order codes", when_to_use="an order code",
                                    content="Channel letter first."), lineage=())
    closer = admit(fx, fields=fields(signature="topic:b", title="Order codes by channel",
                                     when_to_use="an order code is asked for by chat",
                                     content="Channel letter, then the digits."), lineage=())
    await vectors_settled(lessons)
    work = candidate(4, topic="purchase reference by email")
    work.title, work.concern = "Purchase reference by email", "purchase reference by chat"
    _, ids, _ = await lessons.task_lessons(work)
    assert ids == [closer.id, close.id]


async def test_the_owner_turn_gets_the_one_most_related_lesson_by_meaning(fx, embedder):
    lessons = fx.mind.lessons
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    crate_lesson(fx)
    await vectors_settled(lessons)
    text, ids = await lessons.turn_lesson(PARAPHRASE, session_id="day-04")
    assert ids == [lesson.id] and text == lesson.section()
    [note] = [row for row in fx.store.intentions(kind=["note"], limit=50) if row.type == "lesson_use"]
    assert note.context["match"] == "meaning"
    assert await lessons.turn_lesson("What time is the ferry on Friday?", session_id="day-04") == ("", [])


async def test_the_turn_reuses_the_recall_lanes_vector(fx, embedder):
    lessons = fx.mind.lessons
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    with turn_budget.turn(TurnBudget(wait_s=7.5)), source_vectors.turn_query_vectors():
        recalled = await source_vectors.query_vector(embedder, PARAPHRASE, 1.0)       # the recall's vector lane
        text, ids = await lessons.turn_lesson(PARAPHRASE, session_id="day-04")
    assert ids == [lesson.id] and recalled == concept_vector(PARAPHRASE)
    assert embedder.queries == [PARAPHRASE]                                            # one embedding for the turn
    # The memo ends with the turn: the next turn embeds again.
    with source_vectors.turn_query_vectors():
        await lessons.turn_lesson(PARAPHRASE, session_id="day-05")
    assert embedder.queries == [PARAPHRASE, PARAPHRASE]


async def test_the_recall_vector_lane_itself_goes_through_the_turns_memo(tmp_path, embedder):
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.vector.indexes import IndexCatalog
    from colonymind.vector.store import VectorStore
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    store = VectorStore(str(tmp_path / "lancedb"), identity=embedder.index_identity, catalog=IndexCatalog(ledger))
    await store.connect(WIDTH)
    await store.ensure_collections(WIDTH)
    projection = source_vectors.SourceVectors(ledger, store, embedder)
    ledger.record_source("t-1", contact_id="c", session_id="s", messages=[{"role": "user", "content": "An order."}])
    while await projection.process_one():
        pass
    with source_vectors.turn_query_vectors():
        await projection.search("the purchase reference", contact_id="c", session_id="s")
        await source_vectors.query_vector(embedder, "the purchase reference", 1.0)
    assert embedder.queries == ["the purchase reference"]


async def test_without_the_recall_vector_one_embed_call_within_the_turns_budget(fx, embedder, monkeypatch):
    lessons = fx.mind.lessons
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    embedder.delay = 0.5
    monkeypatch.setenv(turn_budget.EMBED_ENV, "50")                 # a 50 ms attempt
    started = time.monotonic()
    with turn_budget.turn(TurnBudget(wait_s=7.5)), source_vectors.turn_query_vectors():
        assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert time.monotonic() - started < 0.4
    assert embedder.queries == [PARAPHRASE]
    assert lessons.match_counts()[match.UNAVAILABLE] == 1
    # A turn with no budget left starts no call.
    with turn_budget.turn(TurnBudget(wait_s=0.0)), source_vectors.turn_query_vectors():
        assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert embedder.queries == [PARAPHRASE] and lessons.match_counts()[match.UNAVAILABLE] == 2


async def test_an_embedding_the_recall_lane_gave_up_on_is_abandoned_and_never_tried_again_in_the_turn(fx, embedder):
    """When the last caller waiting on the turn's query embedding gives up, the call is abandoned at once (not left
    running to the turn's end), and the lesson match that asks later gets nothing by meaning without a second call."""
    lessons = fx.mind.lessons
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    embedder.delay = 0.5
    with turn_budget.turn(TurnBudget(wait_s=7.5)), source_vectors.turn_query_vectors():
        with pytest.raises(TimeoutError):
            await source_vectors.query_vector(embedder, PARAPHRASE, 0.05)           # the recall lane gives up
        await asyncio.sleep(0)
        assert embedder.abandoned == 1
        assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert embedder.queries == [PARAPHRASE] and lessons.match_counts()[match.UNAVAILABLE] == 1


async def test_without_an_embedder_or_with_another_identity_nothing_by_meaning_and_it_is_counted(fx, embedder,
                                                                                                    monkeypatch):
    lessons = fx.mind.lessons
    own = admit(fx, fields=fields(), lineage=())
    admit(fx, fields=fields(signature="topic:order-codes-2"), lineage=())
    await vectors_settled(lessons)
    monkeypatch.setattr(vector_module, "_pipeline", None)
    assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    _, ids, kinds = await lessons.task_lessons(candidate(1))
    assert ids == [own.id] and kinds == {own.id: "own_class"}                          # own class as before
    other = ConceptEmbedder("another/model")
    monkeypatch.setattr(vector_module, "_pipeline", other)
    assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert other.queries == [] and other.batches == []                                 # nothing on the owner's path
    assert lessons.match_counts()[match.UNAVAILABLE] == 3


async def test_an_embedding_measured_below_the_floor_gets_nothing_by_meaning(fx, embedder, monkeypatch):
    small = shipped("Qwen/Qwen3-Embedding-0.6B")["embedding"]
    embedder.index_identity = EmbeddingIdentity(**{key: value for key, value in small.items()
                                                   if key != "fingerprint"})
    lessons = fx.mind.lessons
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert embedder.queries == []                                                      # no call for no bound


# -- the lesson vectors ---------------------------------------------------------------------------

def stored(fx):
    path = fx.mind.state_dir / match.VECTORS_FILE
    return json.loads(path.read_text()) if path.exists() else None


async def test_a_lesson_vector_is_written_at_admission_as_a_document(fx, embedder):
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(fx.mind.lessons)
    data = stored(fx)
    assert data["fingerprint"] == embedder.index_identity.fingerprint and list(data["vectors"]) == [lesson.id]
    assert data["vectors"][lesson.id]["hash"] == match.text_hash(match.lesson_text(lesson.title, lesson.when_to_use))
    assert embedder.batches == [[match.lesson_text(lesson.title, lesson.when_to_use)]] and embedder.queries == []
    assert oct((fx.mind.state_dir / match.VECTORS_FILE).stat().st_mode & 0o777) == "0o600"


async def test_a_relayed_lesson_gets_its_vector(fx, embedder):
    lesson, created = fx.mind.lessons.relay({"kind": "strategy", "title": "Order codes by channel",
                                             "when_to_use": "an order code is asked for",
                                             "content": "Start with the channel letter."},
                                            said_at=fx.now.isoformat(), note="said on a call", recorded_by="operator")
    await vectors_settled(fx.mind.lessons)
    assert created and list(stored(fx)["vectors"]) == [lesson.id]


async def test_a_cooled_lesson_loses_its_vector_and_a_revived_one_gets_it_back(tmp_path, monkeypatch, embedder):
    from test_mind_lessons import Fixture, OWNER, ROOMY
    from colonymind.api.routers import mind as mind_router
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path, config={**ROOMY, "faculties": {"learning": True}})
    try:
        lessons = fixture.mind.lessons
        lesson = admit(fixture, fields=fields(signature="topic:order-codes"), lineage=())
        await vectors_settled(lessons)
        fixture.shift(days=lessons_module.COLD_DAYS + 1)
        assert lessons.review()["cooled"] == [lesson.id]
        assert stored(fixture) is None
        lessons.witness(lesson.id, evidence=["turn:x"], lineage=[], verified="owner", origin="night")
        await vectors_settled(lessons)
        assert list(stored(fixture)["vectors"]) == [lesson.id]
    finally:
        mind_router.set_mind(None)
        fixture.store.close()


async def test_a_new_embedding_identity_drops_the_vectors_and_the_night_refills_them(fx, embedder, monkeypatch):
    lessons = fx.mind.lessons
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    other = ConceptEmbedder("another/model")
    monkeypatch.setattr(vector_module, "_pipeline", other)
    await lessons.fill_vectors()
    data = stored(fx)
    assert data["fingerprint"] == other.index_identity.fingerprint and list(data["vectors"]) == [lesson.id]
    assert other.batches == [[match.lesson_text(lesson.title, lesson.when_to_use)]]
    # The vectors are a cache: losing the file costs one batch, and the lessons are unchanged.
    (fx.mind.state_dir / match.VECTORS_FILE).unlink()
    lessons.vectors = match.LessonVectors(fx.mind.state_dir / match.VECTORS_FILE)
    await lessons.fill_vectors()
    assert list(stored(fx)["vectors"]) == [lesson.id] and lessons.get(lesson.id) == lesson


async def test_a_lesson_without_a_vector_is_skipped_on_the_owners_turn_with_no_embed_call(fx, embedder,
                                                                                       monkeypatch):
    lessons = fx.mind.lessons
    monkeypatch.setattr(vector_module, "_pipeline", None)       # admitted while no embedder answers: no vector
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    monkeypatch.setattr(vector_module, "_pipeline", embedder)
    with turn_budget.turn(TurnBudget(wait_s=7.5)), source_vectors.turn_query_vectors():
        assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert embedder.queries == [] and all(lesson.id not in batch for batch in embedder.batches)


# -- diagnostics ----------------------------------------------------------------------------------

async def test_each_use_records_how_it_matched_and_the_route_shows_it_with_the_unavailable_count(served, embedder):
    from httpx import ASGITransport, AsyncClient
    from onekey import AUTH
    from colonymind.api.routers import mind as mind_router
    app, fx = served
    app.include_router(mind_router.router)
    own = admit(fx, fields=fields(), lineage=())
    related = admit(fx, fields=fields(signature="topic:order-codes-2", title="Purchase references by channel",
                                      when_to_use="a purchase reference is needed by email",
                                      content="The channel letter comes first."), lineage=())
    await vectors_settled(fx.mind.lessons)
    row = await fx.mind._form(candidate(1), 0.9, fx.now)
    assert row.context["lesson_match"] == {own.id: "own_class", related.id: "meaning"}
    await fx.mind.lessons.turn_lesson(PARAPHRASE, session_id="day-04")
    fx.mind.lessons.count_unavailable()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        value = (await client.get("/v1/mind/lessons", params={"uses": "true"}, headers=AUTH)).json()
    by_kind = sorted((use["kind"], use["lesson_id"], use["match"]) for use in value["uses"])
    assert by_kind == sorted([("task", own.id, "own_class"), ("task", related.id, "meaning"),
                              ("turn", related.id, "meaning")])
    assert value[match.UNAVAILABLE] == 1
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        plain = (await client.get("/v1/mind/lessons", headers=AUTH)).json()
    assert plain["uses"] == [] and match.UNAVAILABLE not in plain


async def test_a_reflectors_row_keeps_no_match_kinds_since_it_records_no_use(fx, embedder):
    """A reflector investigates its class's lessons and does not apply them, so its row names no use: nor how one
    matched, even when the lessons were chosen for it earlier in the tick (``_act``)."""
    own = admit(fx, fields=fields(), lineage=())
    await vectors_settled(fx.mind.lessons)
    reflector = candidate(1, reflector={"request": "Investigate why the order codes fail."})
    await fx.mind._lessons_for(reflector)
    assert reflector.extra == {"lesson_match": {own.id: "own_class"}}
    row = await fx.mind._form(reflector, 0.9, fx.now)
    assert row.lesson_ids is None and "lesson_match" not in row.context


async def test_the_owner_turn_section_comes_by_meaning_and_a_guest_gets_none(served, embedder):
    app, fx = served
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(fx.mind.lessons)
    owner = await sections(app, fx.mind.owner_id, PARAPHRASE, session="day-04")
    assert f"[lesson {lesson.id}," in owner["colonymind-lessons"]["body"]
    assert "colonymind-lessons" not in await sections(app, GUEST, PARAPHRASE, session="day-04")


# -- one rule --------------------------------------------------------------------------------------

def test_the_rule_comparison_is_per_embedder_and_recomputes_from_the_scores():
    """The bounds file's ``choice``: what each rule found at the floor on each embedder measured, recomputed here from
    the dev set (lexical: ``lessons.relevant``, the earlier selection rule) and each measurement's scores (the union at
    its best threshold). It is compared embedder by embedder, never summed: meaning found more than lexical on
    Qwen3-Embedding-8B; on Qwen3-Embedding-0.6B no rule kept the floor, so all three found none; no rule found more
    than meaning on either."""
    choice = json.loads(match.BOUNDS.read_text())["choice"]
    pairs = {str(pair["id"]): pair for pair in match.load_dev_set()}
    lexical = {ident: lessons_module.relevant(SimpleNamespace(**pair["lesson"]), match.pair_texts(pair)[1])
               for ident, pair in pairs.items()}

    def found(hits):
        """The related pairs a rule found, or 0 when under the floor share of what it found is related."""
        chosen = [ident for ident, hit in hits.items() if hit]
        related = sum(pairs[ident]["related"] for ident in chosen)
        return related if chosen and related / len(chosen) >= match.PRECISION_FLOOR else 0
    for entry in json.loads(match.BOUNDS.read_text())["measurements"]:
        scores = {row["id"]: row["score"] for row in entry["pairs"]}
        union = max(found({ident: scores[ident] >= threshold or lexical[ident] for ident in scores})
                    for threshold in set(scores.values()))
        meaning = entry["at_bound"]["tp"] if entry["floor_met"] else 0
        assert choice["found_at_floor"][entry["embedding"]["requested_model"]] == {
            "lexical": found(lexical), "meaning": meaning, "meaning_or_lexical": union}
    counts = choice["found_at_floor"]
    assert counts["Qwen/Qwen3-Embedding-8B"]["meaning"] > counts["Qwen/Qwen3-Embedding-8B"]["lexical"]
    assert counts["Qwen/Qwen3-Embedding-0.6B"] == {"lexical": 0, "meaning": 0, "meaning_or_lexical": 0}
    assert choice["rule"] == "meaning" and all(found_by["meaning"] == max(found_by.values())
                                               for found_by in counts.values())


def test_only_the_measured_rule_remains():
    """Selection consults meaning alone (the bounds file's ``choice``): no lexical relevance in a task's or a turn's
    selection."""
    assert json.loads(match.BOUNDS.read_text())["choice"]["rule"] == "meaning"
    for method in (lessons_module.Lessons.for_task, lessons_module.Lessons.for_turn,
                   lessons_module.Lessons.select_task, lessons_module.Lessons.task_lessons,
                   lessons_module.Lessons.turn_lesson):
        source = inspect.getsource(method)
        assert "relevan" not in source and "MIN_SHARED" not in source and "terms(" not in source, method


async def test_a_lesson_whose_words_match_but_whose_meaning_does_not_is_not_served(fx, embedder):
    """What the lexical rule served (every title and when-to-use term shared) is not served below the bound."""
    lessons = fx.mind.lessons
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    words = "order code asked channel keyboard keyboard keyboard typing typing typing typing"
    assert await lessons.turn_lesson(words, session_id="day-04") == ("", [])


# -- the lesson vectors' erasure is durable ------------------------------------------------

def refuse(name):
    def refused(*args, **kwargs):
        raise PermissionError(f"{name} refused")
    return refused


def test_a_failed_vectors_write_is_retried_by_the_next_write_and_pending_until_then(tmp_path, monkeypatch):
    """Memory already holds the pruned vectors when the file cannot be replaced or removed: the next write retries
    the file although nothing changed in memory, and until it lands the cleanup is pending."""
    path = tmp_path / match.VECTORS_FILE
    vectors = match.LessonVectors(path)
    vectors.write({"a": "ha", "b": "hb"}, fingerprint="fp", add={"a": ("ha", [1.0, 0.0]), "b": ("hb", [0.0, 1.0])})
    assert not vectors.pending
    with monkeypatch.context() as patched:
        patched.setattr(match.os, "replace", refuse("replace"))
        with pytest.raises(PermissionError):
            vectors.write({"a": "ha"})                                  # b is erased; the file cannot be replaced
    assert vectors.pending and set(json.loads(path.read_text())["vectors"]) == {"a", "b"}
    assert vectors.write({"a": "ha"}) is True                           # the same state again: the file is retried
    assert not vectors.pending and set(json.loads(path.read_text())["vectors"]) == {"a"}
    with monkeypatch.context() as patched:
        patched.setattr(type(path), "unlink", refuse("unlink"))
        with pytest.raises(PermissionError):
            vectors.write({})                                           # the last vector's erasure removes the file
    assert vectors.pending and path.exists()
    assert vectors.write({}) is True and not path.exists() and not vectors.pending


# -- a bound belongs to the whole embedding identity ----------------------------------------

def test_a_measurement_belongs_to_the_embedding_that_made_its_vectors(monkeypatch, tmp_path):
    """Another served model, revision, quantization or document format is another embedding: unmeasured. The
    normalization is not part of it, since a cosine does not change with a vector's length."""
    monkeypatch.delenv(match.DEPLOYMENT_BOUNDS, raising=False)
    large = shipped("Qwen/Qwen3-Embedding-8B")
    for changed in ({"served_model": "other/model"}, {"declared_revision": "0" * 40}, {"declared_revision": "unknown"},
                    {"quantization": "int8"}, {"document_format": "prefix-v1:passage: "}):
        assert match.bound_for({**large["embedding"], **changed}) == (match.DEFAULT_BOUND, "unmeasured"), changed
    assert match.bound_for({**large["embedding"], "normalization": "provider-l2"}) == (large["bound"], "measured")
    # Recording another revision's measurement keeps the first one beside it.
    own = tmp_path / "bounds.json"
    match.record(large, own)
    match.record({**large, "embedding": {**large["embedding"], "declared_revision": "0" * 40}, "bound": 0.6}, own)
    assert len(json.loads(own.read_text())["measurements"]) == 2


# -- the turn's shared query embedding ------------------------------------------------------

async def test_a_caller_right_after_the_last_waiter_gave_up_gets_a_timeout_not_a_cancellation(embedder):
    """The abandoned call is not cancelled yet when the next caller asks (no yield between): it is refused as a
    timeout, and a caller that is itself cancelled still sees its own cancellation."""
    embedder.delay = 0.5
    with source_vectors.turn_query_vectors():
        with pytest.raises(TimeoutError):
            await source_vectors.query_vector(embedder, PARAPHRASE, 0.05)
        with pytest.raises(TimeoutError):
            await source_vectors.query_vector(embedder, PARAPHRASE, 1.0)
    assert embedder.queries == [PARAPHRASE]
    with source_vectors.turn_query_vectors():
        waiting = asyncio.ensure_future(source_vectors.query_vector(embedder, "another text", 5.0))
        await asyncio.sleep(0.01)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    # A caller still waiting when the turn ends (its call abandoned with the turn) gets a timeout too.
    with source_vectors.turn_query_vectors():
        straggler = asyncio.ensure_future(source_vectors.query_vector(embedder, "a third text", 5.0))
        await asyncio.sleep(0.01)
    with pytest.raises(TimeoutError):
        await straggler


async def test_the_lesson_match_right_after_recall_gave_up_is_unavailable_not_a_cancelled_turn(fx, embedder):
    lessons = fx.mind.lessons
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(lessons)
    embedder.delay = 0.5
    with turn_budget.turn(TurnBudget(wait_s=7.5)), source_vectors.turn_query_vectors():
        with pytest.raises(TimeoutError):
            await source_vectors.query_vector(embedder, PARAPHRASE, 0.05)           # the recall lane gives up
        assert await lessons.turn_lesson(PARAPHRASE, session_id="day-04") == ("", [])
    assert embedder.queries == [PARAPHRASE] and lessons.match_counts()[match.UNAVAILABLE] == 1


# -- one fill of the lesson vectors, cancelled with the night -------------------------------

class SlowBatches(ConceptEmbedder):
    """Each batch takes ``batch_delay`` seconds."""

    batch_delay = 0.2

    async def embed_batch(self, texts):
        self.batches.append(list(texts))
        await asyncio.sleep(self.batch_delay)
        return [self.vector(text) for text in texts]


@pytest.fixture
def slow(monkeypatch):
    pipeline = SlowBatches()
    monkeypatch.setattr(vector_module, "_pipeline", pipeline)
    monkeypatch.delenv(match.DEPLOYMENT_BOUNDS, raising=False)
    return pipeline


async def test_the_nights_fill_joins_the_one_an_admission_started(fx, slow):
    lesson = admit(fx, fields=fields(signature="topic:order-codes"), lineage=())    # starts a fill
    assert await fx.mind.lessons.fill_vectors() == 1                                # the night's: the same one
    assert slow.batches == [[match.lesson_text(lesson.title, lesson.when_to_use)]]
    assert list(stored(fx)["vectors"]) == [lesson.id]


async def test_a_fill_is_cancelled_with_the_night_and_by_the_off_switch(fx, slow):
    lessons = fx.mind.lessons
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())             # starts a fill
    night = asyncio.ensure_future(lessons.fill_vectors())                           # the night joins it
    await asyncio.sleep(0.05)
    night.cancel()
    with pytest.raises(asyncio.CancelledError):
        await night
    await asyncio.sleep(0.3)
    assert stored(fx) is None and len(slow.batches) == 1
    crate_lesson(fx)                                                                # another fill starts
    await asyncio.sleep(0.05)
    fx.mind.off(reason="owner")
    await asyncio.sleep(0.3)
    assert stored(fx) is None and len(slow.batches) == 2


# -- the off switch during the owner turn's embedding ---------------------------------------

async def test_an_owner_turn_whose_embedding_outlives_the_off_switch_gets_no_lesson(served, embedder):
    from colonymind.api.routers import host
    app, fx = served
    admit(fx, fields=fields(signature="topic:order-codes"), lineage=())
    await vectors_settled(fx.mind.lessons)
    embedder.delay = 0.2
    with turn_budget.turn(TurnBudget(wait_s=7.5)), source_vectors.turn_query_vectors():
        turn = asyncio.ensure_future(host._mind_lessons(PARAPHRASE, viewer_is_owner=True, owner_turn=True,
                                                        session_id="day-04"))
        await asyncio.sleep(0.05)
        fx.mind.off(reason="owner")
        assert await turn == ""
    assert not [row for row in fx.store.intentions(kind=["note"], limit=50) if row.type == "lesson_use"]


# -- the default bound and a deployment's measurements --------------------------------------

def test_the_default_bound_is_no_looser_than_any_measured_bound_a_deployments_included(tmp_path, monkeypatch):
    small = shipped("Qwen/Qwen3-Embedding-0.6B")
    own = tmp_path / "bounds.json"
    match.record({**small, "precision_floor": 0.9, "bound": 0.9, "floor_met": True, "at_bound": None}, own)
    monkeypatch.setenv(match.DEPLOYMENT_BOUNDS, str(own))
    assert match.bound_for(small["embedding"]) == (0.9, "measured")
    assert match.bound_for({**small["embedding"], "requested_model": "other/model"}) == (0.9, "unmeasured")
    assert match.bound_for(None) == (0.9, "unmeasured")
    monkeypatch.delenv(match.DEPLOYMENT_BOUNDS)
    assert match.bound_for(None) == (match.DEFAULT_BOUND, "unmeasured")


from colonymind.memory import curated
async def test_a_retired_superseded_or_erased_lesson_loses_its_vector_in_the_same_call(fx, embedder):
    from test_mind_lessons import owner_turn
    lessons = fx.mind.lessons
    owner_turn(fx, 'verdict-1', 'Verdict: the code starts with the channel letter.')
    first = admit(fx)
    other = crate_lesson(fx)
    await vectors_settled(lessons)
    assert set(stored(fx)['vectors']) == {first.id, other.id}
    lessons.retire(other.id, reason='the owner retired it')
    assert set(stored(fx)['vectors']) == {first.id}
    second = admit(fx, fields={'signature': 'topic:order-codes', 'kind': 'strategy', 'title': 'Order codes', 'when_to_use': 'an order code is asked for', 'content': 'Digits first.'})
    assert second.supersedes == first.id
    assert stored(fx) is None
    await vectors_settled(lessons)
    assert set(stored(fx)['vectors']) == {second.id}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, fx.mind.owner_id, source_ids=['verdict-1'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    lessons.prune_vectors()
    assert lessons.get(second.id) is None and stored(fx) is None
