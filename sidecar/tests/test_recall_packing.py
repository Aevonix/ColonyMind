"""Mandatory context caps and bounded reranking."""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.beliefs.source_time import interpret_time_query
from colonymind.memory import compass, recall, recall_stats, selection, turn_budget
from colonymind.memory import trace as recall_trace
from colonymind.memory.recall import (
    calibration_fingerprint, pack_memory_context, provider_calibration_metadata, source_candidates)
from colonymind.memory.search import CollectedSources
from colonymind.memory.selection import RecallSelector
from colonymind.memory.turn_budget import TurnBudget
from colonymind.turns import TurnIdempotencyLedger
from recall_route import ANSWER, OWNER, Reranker, recall_app

QUERY = "harbour timetable"


class Scores:
    """A reranker that scores each document by ``score`` (every one 0.9 by default) and keeps what it was sent."""

    def __init__(self, score=lambda document: 0.9):
        self.score, self.sent = score, []

    def calibration_metadata(self):
        return {"model": "packing-fixture", "format_version": "v1"}

    async def rerank(self, query, documents, top_k=10):
        self.sent.append(list(documents))
        return [{"index": i, "score": self.score(document)} for i, document in enumerate(documents)]


async def failing(query, documents, top_k=10):
    raise RuntimeError("the reranker failed")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_TIMEOUT_MS", "2000")
    monkeypatch.delenv(recall_stats.FALLBACK_ENV, raising=False)
    for name in ("COLONYMIND_RERANKER_REVISION", "COLONYMIND_EMBED_MODEL", "COLONYMIND_EMBED_DIMS",
                 "COLONYMIND_RECALL_INDEX_GENERATION", "COLONYMIND_RECALL_RERANK_TIE_MARGIN"):
        monkeypatch.delenv(name, raising=False)
    recall_stats.STATS.reset()


def calibrated(monkeypatch, reranker, cutoff=".5"):
    """A selector whose cutoff is verified: the configured fingerprint is the reranker's own."""
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", cutoff)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION",
                       calibration_fingerprint(provider_calibration_metadata(reranker)))
    return RecallSelector(reranker.rerank, calibration_metadata=lambda: provider_calibration_metadata(reranker))


def candidates(sessions, *, words=""):
    """Quotations in fused (lexical) order, one per entry of ``sessions`` (the session it was said in)."""
    return [{"id": f"q-{i}", "kind": "source_quote", "epistemic_state": "quotation", "role": "user", "scope": "person",
             "contact_id": "p", "session_id": session, "content": f"Harbour timetable note {i}.{words}",
             "relevance": 1 / (60 + i), "_lexical_rank": i} for i, session in enumerate(sessions)]


def sessions_of(documents, rows):
    by_text = {row["content"]: row["session_id"] for row in rows}
    return [by_text[document] for document in documents]


async def in_turn(coroutine):
    budget = TurnBudget(wait_s=7.5)
    with turn_budget.turn(budget):
        return await coroutine


# --- 1. and 2. The submission: at most 20, spread across sessions -------------------------------------------------

@pytest.mark.parametrize("sessions", [["crowded"] * 15 + ["other"] * 5, ["crowded"] * 12, ["a", "b"] * 10])
async def test_twenty_or_fewer_candidates_are_submitted_as_before(monkeypatch, sessions):
    reranker = Scores()
    rows = candidates(sessions)
    await calibrated(monkeypatch, reranker).select_context(QUERY, [], rows, limit=None)
    assert reranker.sent == [[row["content"] for row in rows]]          # every one, in fused order


async def test_a_crowded_session_no_longer_excludes_the_others(monkeypatch):
    # 24 overlapping chunks of one conversation rank first; six other conversations hold one candidate each. The
    # first 20 in fused order were all the crowded one's. Round robin across sessions sends each of the six, and the
    # crowded one fills the rest with its strongest, in fused order.
    reranker = Scores(lambda document: 0.4 if "note 3." in document else 0.9)
    rows = candidates(["crowded"] * 24 + ["b", "c", "d", "e", "f", "g"])
    selected, _ = await calibrated(monkeypatch, reranker).select_context(QUERY, [], rows, limit=None)
    sent, = reranker.sent
    assert len(sent) == selection.RERANK_SUBMISSION_LIMIT == 20
    assert sent == [row["content"] for row in rows[:14] + rows[24:]]
    assert sessions_of(sent, rows) == ["crowded"] * 14 + ["b", "c", "d", "e", "f", "g"]
    # It reorders only: every score is the reranker's own, and the cutoff still decides.
    assert len(selected) == 19  # every submitted candidate above the verified cutoff survives
    assert all(row["rerank_score"] == 0.9 and row["id"] != "q-3" for row in selected)
    assert {row["id"] for row in selected} <= {row["id"] for row in rows[:14] + rows[24:]}


async def test_a_repeat_of_the_request_is_still_sent_after_every_other_candidate(monkeypatch):
    # An exact repeat of the question goes after independent evidence before bounded reranking. Alone in its
    # session, it would win a round of the spread; it does not: it is sent only when every other candidate is.
    reranker = Scores()
    rows = candidates(["crowded"] * 24 + ["b"])
    repeat = dict(rows[-1], id="q-repeat", session_id="asked-before", content=QUERY)
    await calibrated(monkeypatch, reranker).select_context(QUERY, [], [repeat, *rows], limit=None)
    sent, = reranker.sent
    assert QUERY not in sent and len(sent) == 20
    assert sessions_of(sent, rows) == ["crowded"] * 19 + ["b"]


class Claims:
    """A claim extractor that states one property per sentence of a message (at most six a message)."""
    model = "packing-fixture"

    def __init__(self, outputs):
        self.outputs = outputs

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if kwargs.get("context", {}).get("task") == "source_claim_review":
            return SimpleNamespace(content=json.dumps({str(row["index"]): {
                "keep": True, "reason": "The fixture accepts every grounded proposal."}
                for row in payload["proposals"]}), model_id=self.model)
        return SimpleNamespace(content=json.dumps(self.outputs.get(payload["message"], [])), model_id=self.model)


async def projected(tmp_path):
    """Four long messages of one conversation, each stating six properties (24 assertion bundles), and a short
    answer from another conversation, as the projection prepares them for ranking."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    filler = " The timber decking and the railings are old, and the season is long." * 30
    outputs = {}
    for turn in range(4):
        sentences = [f"Harbour pier {turn * 6 + n} is closed {turn * 6 + n} for repairs." for n in range(6)]
        message = " ".join(sentences) + filler                  # over 2,000 characters: claims, not the message
        outputs[message] = [{
            "subject": "harbour", "predicate": f"pier_{turn * 6 + n}_status", "value": f"closed {turn * 6 + n}",
            "evidence": sentence, "operation": "assert", "prior_claim_id": None, "memory_kind": "personal_context",
            "recall_reason": "Use it when planning a visit.", "valid_from_text": None, "valid_to_text": None,
            "event_at_text": None} for n, sentence in enumerate(sentences)]
        ledger.record_source(f"t-long-{turn}", contact_id="p", session_id="crowded", derive_claims=True,
                             messages=[{"role": "user", "content": message}], occurred_at="2026-03-01T12:00:00+00:00")
    model = Claims(outputs)
    while await projection.process_one(model):
        pass
    ledger.record_source("t-answer", contact_id="p", session_id="other", derive_claims=False,
                         messages=[{"role": "user", "content": "The harbour pier ferry leaves Friday at nine."}],
                         occurred_at="2026-03-02T12:00:00+00:00")
    scope = {"contact_id": "p", "session_id": "now"}
    hits = ledger.search_sources("harbour pier", **scope, limit=40)
    beliefs, rows = projection.prepare_context([], hits, **scope, time_query=interpret_time_query(
        "harbour pier", now=datetime.fromisoformat("2026-03-05T12:00:00+00:00")))
    bundles = [row for row in rows if row.get("content_format") == "source_assertions_v1"]
    # Each bundle names the session of the source it is anchored in (a bundle is grouped by its key, not by a
    # conversation).
    assert len(bundles) == 24 and {row["session_id"] for row in bundles} == {"crowded"}
    assert rows.index(next(row for row in rows if row.get("source_turn_id") == "t-answer")) >= 24
    return beliefs, rows


@pytest.mark.parametrize("excerpts", [True, False])
async def test_projected_assertions_are_spread_by_the_conversation_they_come_from(monkeypatch, tmp_path, excerpts):
    # A projected assertion bundle is grouped by its key, not by a conversation: without a session each one was a
    # conversation of its own and the 24 of them filled the batch ahead of the other conversation's answer. It is
    # spread by the session of the source it is anchored in, which it names, and, with only the bundles left, by
    # that alone; its text is sent as it was.
    beliefs, rows = await projected(tmp_path)
    if not excerpts:                                            # only the bundles speak for the crowded conversation
        rows = [row for row in rows if row.get("content_format") == "source_assertions_v1"
                or row.get("source_turn_id") == "t-answer"]
    reranker = Scores()
    await calibrated(monkeypatch, reranker).select_context("harbour pier", beliefs, rows, limit=None)
    sent, = reranker.sent
    assert len(sent) == 20 and any("Friday at nine" in document for document in sent)
    texts = {row.get("ranking_text", row["content"])[:recall.RERANK_DOCUMENT_CHARS] for row in rows}
    assert set(sent) <= texts                                   # the reranker's text is unchanged


@pytest.mark.parametrize("mode", ["missed", "shadow", "off"])
async def test_without_a_scored_rerank_the_candidates_keep_their_fused_order(monkeypatch, mode):
    # The spread chooses what the reranker reads, nothing else: a rerank that missed, ran in shadow or is off
    # selects from the fused order as before.
    rows = candidates(["crowded"] * 24 + ["b", "c"])
    selector = calibrated(monkeypatch, Scores())
    if mode == "missed":
        selector._rerank_fn = failing
    else:
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK", mode)
    selected, _ = await selector.select_context(QUERY, [], rows, limit=None)
    assert [row["id"] for row in selected] == [f"q-{i}" for i in range(5)]


# --- Mandatory record cap after a verified cutoff --------------------------------------------

async def test_the_default_record_cap_preserves_all_seven_verified_records(monkeypatch):
    selected, content = await calibrated(monkeypatch, Scores()).select_context(
        QUERY, [], candidates(["s"] * 7), limit=None, max_chars=4000)
    assert [row["id"] for row in selected] == [f"q-{i}" for i in range(7)]
    assert len(content) <= 4000 and all(row["rerank_status"] == "scored" for row in selected)


async def test_a_limit_the_caller_names_still_bounds_the_packet(monkeypatch):
    # An explicit search names its own limit (its response carries at most 20 records).
    selected, _ = await calibrated(monkeypatch, Scores()).select_context(
        QUERY, [], candidates(["s"] * 7), limit=5, max_chars=4000)
    assert len(selected) == 5


@pytest.mark.parametrize("case", ["mismatch", "no_cutoff", "shadow"])
async def test_with_the_cutoff_unverified_five_are_selected(monkeypatch, case):
    reranker = Scores()
    selector = calibrated(monkeypatch, reranker)
    if case == "mismatch":
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", "0" * 64)
    elif case == "no_cutoff":
        monkeypatch.delenv("COLONYMIND_RECALL_RERANK_MIN_SCORE")
    else:
        monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "shadow")
    selected, _ = await selector.select_context(QUERY, [], candidates(["s"] * 7), limit=None, max_chars=4000)
    assert len(selected) == recall.MEMORY_PACKET_LIMIT == 5


@pytest.mark.parametrize("bound, count", [("", 5), ("3", 3), ("0", 0)])
async def test_on_a_missed_rerank_the_fallback_limit_is_unchanged(monkeypatch, bound, count):
    monkeypatch.setenv(recall_stats.FALLBACK_ENV, bound)
    selector = calibrated(monkeypatch, Scores())
    selector._rerank_fn = failing
    selected, _ = await in_turn(selector.select_context(QUERY, [], candidates(["s"] * 7), limit=None))
    assert len(selected) == count
    assert recall_stats.STATS.snapshot()["turns"]["fallback_rows"] == count


# --- The character budget and reranker submission bound still apply ---------------------

async def test_the_packet_never_exceeds_its_budget(monkeypatch):
    words = " Timetable detail." * 45                                       # about 800 characters each
    selector = calibrated(monkeypatch, Scores())
    for max_chars in (4000, 10000, 24000):
        selected, content = await selector.select_context(
            QUERY, [], candidates(["s"] * 30, words=words), limit=None, max_chars=max_chars)
        assert 0 < len(content) <= max_chars
        assert 0 < len(selected) <= selection.RERANK_SUBMISSION_LIMIT


async def test_with_packing_active_the_reranker_still_receives_at_most_twenty(monkeypatch):
    reranker = Scores()
    selector = calibrated(monkeypatch, reranker)
    selected, _ = await selector.select_context(QUERY, [], candidates(["s"] * 40), limit=None, max_chars=24000)
    assert [len(batch) for batch in reranker.sent] == [20]
    assert len(selected) == 20                 # the default cap does not truncate verified reranking
    # A caller that names its own limit (an explicit search) sends four times it, as before.
    await selector.select_context(QUERY, [], candidates(["s"] * 40), limit=20, max_chars=24000)
    await selector.select_context(QUERY, [], candidates(["s"] * 40), limit=2, max_chars=24000)
    assert [len(batch) for batch in reranker.sent] == [20, 40, 8]


async def test_a_packet_past_the_limit_is_counted(monkeypatch):
    await in_turn(calibrated(monkeypatch, Scores()).select_context(QUERY, [], candidates(["s"] * 7), limit=None))
    await calibrated(monkeypatch, Scores()).select_context(QUERY, [], candidates(["s"] * 3), limit=None)
    snapshot = recall_stats.STATS.snapshot()
    assert "packed_unlimited" not in snapshot["turns"]
    assert "rows_past_limit" not in snapshot["searches"]


# --- The trace names the new ways out ------------------------------------------------------------------------------

def _ledger(tmp_path, sessions):
    """A ledger with one message per entry of ``sessions``, and its candidates in that order (fused order)."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    for i, session in enumerate(sessions):
        ledger.record_source(f"t-{i:02d}", contact_id="p", session_id=session, derive_claims=False,
                             messages=[{"role": "user", "content": f"Harbour timetable note marker{i:02d}."}])
    hits = [hit for i in range(len(sessions))
            for hit in ledger.search_sources(f"marker{i:02d}", contact_id="p", session_id="s", limit=1)]
    assert [hit["turn_id"] for hit in hits] == [f"t-{i:02d}" for i in range(len(sessions))]
    return ledger, source_candidates(hits)


async def test_the_trace_names_a_candidate_the_spread_left_out(monkeypatch, tmp_path):
    ledger, rows = _ledger(tmp_path, ["crowded"] * 24 + ["b", "c"])
    tracer = recall_trace.RecallTrace()
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=None, max_chars=4000)
    selected, content = await calibrated(monkeypatch, Scores()).select_context(
        QUERY, [], rows, limit=None, max_chars=24000, trace=tracer)
    tracer.packet(selected, content)
    report = tracer.report()
    exits = {row["turn_id"]: row["exit"] for row in report["rows"]}
    assert {turn for turn, exit_ in exits.items() if exit_ is None} == {
        *(f"t-{i:02d}" for i in range(18)), "t-24", "t-25"}
    assert {turn for turn, exit_ in exits.items() if exit_ == "over_session_share"} == {"t-18", "t-19"}
    assert {turn for turn, exit_ in exits.items() if exit_ == "below_submission_bound"} == {
        f"t-{i:02d}" for i in range(20, 24)}
    # The trace reports the mandatory context record cap.
    assert report["packet"] == {"limit": 20, "max_chars": 24000, "count": 20}


def test_the_packer_names_the_hosts_allowance_when_it_left_a_candidate_out(tmp_path):
    ledger, rows = _ledger(tmp_path, ["s"] * 4)
    tracer = recall_trace.RecallTrace()
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=None, max_chars=4000)
    tracer.fused(rows)
    full, _ = pack_memory_context(rows, limit=None, max_chars=4000, trace=tracer)
    assert len(full) == 4
    kept, content = pack_memory_context(full, limit=None, max_chars=900, trace=tracer, over_budget="over_host_allowance")
    tracer.packet(kept, content)
    exits = [row["exit"] for row in tracer.report()["rows"]]
    assert 0 < len(kept) < 4 and exits.count(None) == len(kept)
    assert exits.count("over_host_allowance") == 4 - len(kept)
    assert set(recall_trace.EXITS) >= {"over_session_share", "over_host_allowance"}
    from colonymind import recall_cli
    assert set(recall_cli.EXITS) == set(recall_trace.EXITS)             # every exit has its words


# --- Packing again within the host's allowance ---------------------------------------------------------------------

STEPS = ("Step: open the harbour valve and log the pressure. " * 80)[:3420]


def procedure_and_answer(tmp_path):
    """A long procedure recalled whole, ranked first, and a short answer after it: recall's 4,000-character budget has
    room for the procedure alone."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("t-proc", contact_id="p", session_id="s", derive_claims=False,
                         messages=[{"role": "user", "content": STEPS}])
    ledger.record_source("t-ans", contact_id="p", session_id="s", derive_claims=False,
                         messages=[{"role": "user", "content": "The hydrofoil leaves Friday at nine."}])
    answer, = source_candidates(ledger.search_sources("hydrofoil", contact_id="p", session_id="s", limit=1))
    procedure = {"id": "procedure-source:" + "a" * 64, "kind": "source_quote", "source_uri": "turn:t-proc",
                 "source_turn_id": "t-proc", "source_turn_ids": ["t-proc"], "role": "user", "contact_id": "p",
                 "session_id": "s", "scope": "person", "epistemic_state": "quotation", "atomic_evidence": True,
                 "procedure_context": "complete_source_message_text", "claim_status": "source_procedure_context",
                 "history_anchor": {"source_id": "t-proc", "claim_id": "claim-1"},
                 "source_anchors": [{"source_id": "t-proc"}], "content": STEPS, "ranking_text": STEPS,
                 "relevance": 1.0}
    return ledger, [procedure, answer]


async def offered_packet(monkeypatch, rows, **kwargs):
    with selection.keep_offered() as offered:
        selected, content = await calibrated(monkeypatch, Scores()).select_context(
            QUERY, [], rows, limit=None, max_chars=4000, **kwargs)
    assert [row["id"] for row in selected] == [rows[0]["id"]] and selected[0]["content"] == STEPS
    return selected, content, offered


async def test_an_answer_behind_a_shrinking_procedure_takes_the_room_it_frees(monkeypatch, tmp_path):
    # Packed again from recall's packet, the procedure shrinks to a pointer to its source and the answer, which
    # recall's budget left out behind it, is never considered. Packed again from what recall's packer was offered,
    # the answer fits beside the pointer.
    ledger, rows = procedure_and_answer(tmp_path)
    tracer = recall_trace.RecallTrace()
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=None, max_chars=4000)
    selected, _, offered = await offered_packet(monkeypatch, rows, trace=tracer)
    first, = offered
    assert [row["id"] for row in first.rows] == [row["id"] for row in rows] and first.limit == 20
    kept, content = recall.fit_packet(first.rows, limit=first.limit, max_chars=1400, recalled=selected, trace=tracer)
    tracer.packet(kept, content)
    assert [row["id"] for row in kept] == [row["id"] for row in rows] and len(content) <= 1400
    assert kept[0]["procedure_context"] == "full_source_required" and "Friday at nine" in content
    exits = {row["turn_id"]: row["exit"] for row in tracer.report()["rows"]}
    assert exits == {"t-proc": None, "t-ans": None}


async def test_context_assembly_packs_what_recall_was_offered_within_the_allowance(monkeypatch, tmp_path):
    from colonymind.api.routers import host
    from colonymind.api.schemas.host import ContextSection
    from colonymind.memory.search import MemoryPacket
    from colonymind.memory.trace import NULL_TRACE
    ledger, rows = procedure_and_answer(tmp_path)
    selected, content, offered = await offered_packet(monkeypatch, rows)
    refs = ledger.source_references(["t-proc"], contact_id="p", session_id="s")
    recalled = SimpleNamespace(packet=MemoryPacket(selected, content, refs, None, {}, []), offered=offered[-1],
                               ledger=ledger, contact_id="p", session_id="s", served=None)
    clock = ContextSection(id="temporal-context", title="Current Time", body="Current UTC time: 2026-03-05", priority=100)
    memory = ContextSection(id=compass.RECALLED, title="Relevant Memories", body=content, priority=90, citations=refs)
    allowance = (len(clock.title) + len(clock.body) + 6) + (len(memory.title) + 6) + 1400
    sections, shown = await host._memory_within_allowance([clock, memory], recalled, inline_chars=allowance,
                                                          trace=NULL_TRACE)
    fitted = sections[1]
    assert "Friday at nine" in fitted.body and len(fitted.body) <= 1400
    assert [row["id"] for row in shown] == [row["id"] for row in rows]
    assert {ref["source_id"] for ref in fitted.citations} == {"t-proc", "t-ans"}     # the answer's source is cited


# --- Through the route -----------------------------------------------------------------------------------------------

class Passing(Reranker):
    """Every candidate passes the cutoff; the answer scores highest."""

    async def _answer(self, query, documents):
        self.calls.append({"documents": len(documents)})
        return [{"index": i, "score": 0.99 if "Friday at nine" in doc else 0.9} for i, doc in enumerate(documents)]


async def post(app, path, body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def assemble_body(**changes):
    return {"identity": {"host_id": "fixture"}, "context": {"contact_id": OWNER, "session_id": "session-b"},
            "incoming_message": {"role": "user", "content": "hydrofoil departure"}, "trace": True, **changes}


def memory_of(response):
    return next(section for section in response["sections"] if section["id"] == compass.RECALLED)


async def test_a_turns_packet_keeps_its_record_cap_after_the_verified_cutoff(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=0)
    response = await post(setup.app, "/v1/host/context/assemble", assemble_body())
    memory = memory_of(response)
    assert ANSWER in memory["body"] and memory["body"].count('"kind": "source_quote"') == 8
    assert response["trace"]["packet"]["count"] == 8 and response["trace"]["section"] == "in_context"
    # An explicit search keeps its own limit (five unless it names another).
    search = await post(setup.app, "/v1/host/memory/search", {
        "identity": {"host_id": "fixture"}, "person_id": OWNER, "session_id": "session-b",
        "query": "hydrofoil departure"})
    assert search["count"] == 5


def _cost(section):
    return len(section["title"] or section["id"]) + len(section["body"]) + 6   # as the host renders it


async def test_the_packet_stays_within_the_hosts_allowance_less_the_pinned_items(monkeypatch, tmp_path):
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=0)
    whole = memory_of(await post(setup.app, "/v1/host/context/assemble", assemble_body()))
    allowance = 1200
    assert _cost(whole) > allowance                                    # not vacuous: the whole packet cannot fit
    response = await post(setup.app, "/v1/host/context/assemble", assemble_body(inline_chars=allowance))
    memory = memory_of(response)
    pinned = sum(_cost(section) for section in response["sections"] if section["id"] in compass.PINNED_SECTIONS)
    assert pinned > 0 and ANSWER in memory["body"]
    assert _cost(memory) <= allowance - pinned
    # It reached the context as recall packed it: no block fit shortened or held it back.
    assert response["trace"]["section"] == "in_context"
    assert compass.RECALLED not in {item["id"] for item in response["budget"]["rotated"] + response["budget"]["held_back"]}
    exits = [row["exit"] for row in response["trace"]["rows"]]
    assert exits.count("over_host_allowance") == 8 - response["trace"]["packet"]["count"] > 0
    # Its citations are the sources it still shows.
    cited = {ref["source_id"] for ref in memory["citations"]}
    assert cited and all(f'"turn:{source}"' in memory["body"] for source in cited)
    assert sum(_cost(section) for section in response["sections"]) <= allowance


# Around the smallest record: at 380 and 410 not even the section's heading fits beside the clock, from 440 to 590
# a heading and the packet's first line would, but not one record, and from 620 one record fits.
ALLOWANCES = range(380, 680, 30)


async def test_a_memory_with_no_record_that_fits_is_held_back_and_named(monkeypatch, tmp_path):
    # When not one record fits beside the pinned items, the section is held back whole: never its heading and the
    # packet's first line without a record, never a packet over the allowance, and never unnamed. With the resident
    # blocks on, the budget report and the "Held back this turn" section name it; the trace says so either way.
    from colonymind.memory import blocks
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=0)
    blocks.reset_settings()
    try:
        outcomes = set()
        for allowance in ALLOWANCES:
            response = await post(setup.app, "/v1/host/context/assemble", assemble_body(inline_chars=allowance))
            by_id = {section["id"]: section for section in response["sections"]}
            assert sum(_cost(section) for section in response["sections"]) <= allowance
            memory = by_id.get(compass.RECALLED)
            if memory is not None:
                assert memory["body"].count('"kind": "source_quote"') >= 1 and memory["citations"]
                assert response["trace"]["section"] == "in_context"
                outcomes.add("shown")
                continue
            outcomes.add("held_back")
            assert response["trace"]["section"] == "held_back"
            assert not any(row["exit"] == "over_host_allowance" for row in response["trace"]["rows"])
            assert compass.RECALLED in {item["id"] for item in response["budget"]["held_back"]}
            assert "Relevant Memories" in by_id[blocks.HELD_ID]["body"]
        assert outcomes == {"shown", "held_back"}                      # not vacuous: the boundary is crossed
        # An allowance too small for the words that name it still cannot overflow.
        response = await post(setup.app, "/v1/host/context/assemble", assemble_body(inline_chars=160))
        assert compass.RECALLED in {item["id"] for item in response["budget"]["held_back"]}
        assert compass.RECALLED not in {section["id"] for section in response["sections"]}
        assert sum(_cost(section) for section in response["sections"]) <= 160
    finally:
        blocks.reset_settings()


def superseded(n):
    """A changed claim the conversation was served earlier, and the line it was served in."""
    claim = "claim:" + f"{n:064x}"
    record = compass.Superseded(old=f"the east pier {n}", current=f"the west dock {n}", since="2026-09-18",
                                subject=f"ferry {n}", kind="changed", record=claim)
    line = "- " + json.dumps({"kind": "source_quote", "claim_id": claim, "source": "turn:earlier"}) + " " + json.dumps(
        f"The ferry {n} leaves from the east pier {n}.")
    return record, line


@pytest.fixture
def served_earlier(monkeypatch):
    """Two values this conversation was served on an earlier turn and the record has changed since: with the
    superseded-value annotation on, the turn's context carries their corrections, pinned beside the memory."""
    monkeypatch.setenv(compass.ANNOTATION_ENV, "on")
    records, lines = zip(*(superseded(n) for n in range(2)))
    monkeypatch.setattr(compass, "claim_supersessions", lambda ledger, **scope: list(records))
    compass.SERVED.clear()
    compass.SERVED.remember((OWNER, "session-b"), "\n".join(lines))
    yield records
    compass.SERVED.clear()


async def test_the_allowance_counts_the_corrections_owed_to_earlier_turns(monkeypatch, tmp_path, served_earlier):
    # The corrections to earlier context are pinned and added once the lanes are assembled; the memory is fitted in
    # what is left beside them, so no block fit has to cut the recalled evidence to make room.
    setup = await recall_app(monkeypatch, tmp_path, reranker=Passing(), initiatives=0)
    allowance = 1600                                    # room for the answer beside the clock and the corrections
    response = await post(setup.app, "/v1/host/context/assemble", assemble_body(inline_chars=allowance))
    by_id = {section["id"]: section for section in response["sections"]}
    assert all(record.record in by_id[compass.CORRECTIONS]["body"] for record in served_earlier)
    memory = by_id[compass.RECALLED]
    pinned = sum(_cost(section) for section in response["sections"] if section["id"] in compass.PINNED_SECTIONS)
    assert ANSWER in memory["body"] and _cost(memory) <= allowance - pinned
    assert response["trace"]["section"] == "in_context"
    assert compass.RECALLED not in {item["id"] for item in response["budget"]["rotated"] + response["budget"]["held_back"]}
    assert sum(_cost(section) for section in response["sections"]) <= allowance


def claim_cards(count):
    """Recalled assertion cards, each of a claim the record has changed since: annotated, every card's line grows."""
    records, rows = [], []
    for n in range(count):
        record, _ = superseded(n)
        records.append(record)
        card = {"subject": record.subject, "predicate": "departs_from", "status": "source_assertion", "assertions": [
            {"claim_id": record.record, "source": f"turn:t-{n}", "source_message_hash": f"h{n}", "role": "user",
             "value": record.old, "quote": f"The ferry {n} leaves from {record.old}.",
             "reported_at": "2026-03-01T12:00:00+00:00"}]}
        rows.append({"id": f"assertions:{n:064x}", "kind": "source_quote", "source_uri": f"turn:t-{n}",
                     "source_turn_ids": [f"t-{n}"], "_source_message_hashes": {f"t-{n}": [f"h{n}"]},
                     "epistemic_state": "source_assertion", "atomic_evidence": True,
                     "content_format": "source_assertions_v1", "content": json.dumps(card),
                     "history_anchor": {"source_id": f"t-{n}", "claim_id": record.record}})
    return records, rows


async def test_the_memorys_own_annotations_are_counted_within_the_allowance():
    # With the annotation on, a recalled line stating a changed value carries a note of the current one. The packet
    # has to fit as it is shown, notes included: one that fits the room only without them is packed within less.
    from colonymind.api.routers import host
    from colonymind.api.schemas.host import ContextSection
    from colonymind.memory.search import MemoryPacket
    from colonymind.memory.trace import NULL_TRACE
    records, rows = claim_cards(3)
    selected, content = pack_memory_context(rows, limit=None, max_chars=4000)
    assert len(selected) == 3
    memory = ContextSection(id=compass.RECALLED, title="Relevant Memories", priority=90,
                            body=compass.annotate_superseded([ContextSection(id=compass.RECALLED, body=content)],
                                                             records)[0].body)
    clock = ContextSection(id="temporal-context", title="Current Time", body="Current UTC time: 2026-03-05", priority=100)
    room = len(content) + 50                            # the packet fits without its notes, not with them
    assert len(memory.body) > room
    allowance = (len(clock.title) + len(clock.body) + 6) + (len(memory.title) + 6) + room
    recalled = SimpleNamespace(packet=MemoryPacket(selected, content, [], None, {}, []),
                               offered=selection.Offered(tuple(rows), None, 4000), ledger=None, contact_id="p",
                               session_id="s", served=None)
    sections, shown = await host._memory_within_allowance([clock, memory], recalled, inline_chars=allowance,
                                                          records=records, trace=NULL_TRACE)
    fitted = sections[1].body
    assert len(fitted) <= room and 0 < len(shown) < 3
    assert "[superseded id=" in fitted and compass.dead_value_lines(fitted, records) == 0
