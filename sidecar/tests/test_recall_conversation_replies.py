"""A user message found alone brings its single direct reply (docs/RECALL-HYBRID.md, Returning no useful memory:
"Conversation pairs").

Pairing used to start only from a reply: a found assistant message brought its direct input. A found user message
never brought its reply, so "what did you recommend for the fence" matched the question and lost the answer. Now a
user message resolves its one direct reply, the next message of its own envelope or a later turn whose
``_supplied_inputs`` names it (through the derived ``source_replies`` map, never a scan of the sources), checks it as
the existing pairing does, and recall shows the two as the existing ``conversation_pair``. No reply, or more than one,
is no pair.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.errors import install_exception_handlers
from colonymind.api.routers import host
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.beliefs.source_time import MemoryTimeQuery
from colonymind.memory.search import CollectedSources, select_memory
from colonymind.memory.selection import RecallSelector
from colonymind.turns.idempotency import TurnIdempotencyLedger, source_message_hash
from test_source_claim_projection import Model, claim

SCOPE = {"contact_id": "person", "session_id": "later"}
ASKED = "Which shrubs suit the shady north fence?"
ANSWER = "Hollies do well there; they take the shade and the wind."
USER = {"role": "user", "content": ASKED}


def ledger_at(tmp_path):
    return TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")


def envelope(ledger, turn="turn", *, session="original", reply=ANSWER, contact="person", question=ASKED):
    """The question and its answer in one source."""
    ledger.record_source(turn, contact_id=contact, session_id=session, derive_claims=False,
                         messages=[{"role": "user", "content": question}, {"role": "assistant", "content": reply}])


def asked(ledger, turn="input", *, session="original"):
    """The question in a source of its own."""
    ledger.record_source(turn, contact_id="person", session_id=session, derive_claims=False, messages=[USER])


def answered(ledger, turn="reply", *, parent="input", session="original", reply=ANSWER):
    """A later turn whose answer names the question as its direct input."""
    ledger.record_source(turn, contact_id="person", session_id=session, derive_claims=False, messages=[{
        "role": "assistant", "content": reply,
        "_supplied_inputs": [{"source_id": parent, "input_message_hash": source_message_hash(session, USER)}]}])


SHAPES = {"envelope": lambda ledger: envelope(ledger),
          "later-turn": lambda ledger: (asked(ledger), answered(ledger))}
REPLY_TURN = {"envelope": "turn", "later-turn": "reply"}


def question_hits(ledger):
    """What a keyword lane finds for words only the question holds: the user message alone."""
    hits = ledger.search_sources("shrubs", **SCOPE, limit=20)
    assert [hit["role"] for hit in hits] == ["user"] and hits[0]["content"] == ASKED
    return hits


def prepare(ledger, hits):
    return SourceClaimProjection(ledger).prepare_context([], hits, **SCOPE, time_query=MemoryTimeQuery(),
                                                         include_conversation_inputs=True)[1]


async def packet(ledger, hits, *, query="Which shrubs did you recommend for the fence?"):
    return await select_memory(CollectedSources(ledger, **SCOPE, watermark=ledger.erasure_watermark("person"),
                                                hits=list(hits)),
                               query=query, selector=RecallSelector(), limit=5, timezone_name="UTC")


def reply_rows(ledger):
    from colonymind.turns import source_replies
    with closing(ledger._connect()) as conn:
        return sorted(tuple(row) for row in conn.execute(
            f"SELECT input_turn, input_hash, reply_turn, reply_hash FROM {source_replies.TABLE}"))


# --- 5. A user hit with one reply forms a pair ----------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
async def test_a_user_message_found_alone_brings_its_one_reply(tmp_path, monkeypatch, shape):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = ledger_at(tmp_path)
    SHAPES[shape](ledger)
    hits = question_hits(ledger)
    row, = prepare(ledger, hits)
    assert row["kind"] == "conversation_pair" and row["epistemic_state"] == "quotation"
    pair = json.loads(row["content"])
    assert (pair["input"]["role"], pair["input"]["quote"]) == ("user", ASKED)
    assert (pair["response"]["role"], pair["response"]["quote"]) == ("assistant", ANSWER)
    assert pair["response"]["source_id"] == REPLY_TURN[shape]
    result = await packet(ledger, hits)
    selected, = result.selected
    assert selected["kind"] == "conversation_pair"
    assert ASKED in result.content and ANSWER in result.content
    assert result.content.index(ASKED) < result.content.index(ANSWER)
    for role in ("input", "response"):
        ref = {"source_id": pair[role]["source_id"], "source_version": pair[role]["source_version"]}
        assert ref in result.source_refs


def test_a_reply_is_derived_when_its_source_is_recorded(tmp_path):
    ledger = ledger_at(tmp_path)
    envelope(ledger, "turn")
    asked(ledger, "input")
    answered(ledger, "reply", parent="input")
    question = source_message_hash("original", USER)
    answer = source_message_hash("original", {"role": "assistant", "content": ANSWER})
    assert reply_rows(ledger) == sorted([("input", question, "reply", answer), ("turn", question, "turn", answer)])


# --- 6. An erased, annotated or out-of-scope reply is not attached --------------------------------------------------


def annotate_reply(ledger, shape):
    ref = ledger.source_references([REPLY_TURN[shape]], **SCOPE)[0]
    ledger.append_source_annotation(**SCOPE, annotation_id="owner-note", **ref, excerpt=ANSWER,
                                    correction="Hollies were a guess; the nursery advised ferns.",
                                    author_principal="person")


def sql(statement):
    def change(ledger, shape):
        with closing(sqlite3.connect(ledger.db_path)) as conn, conn:
            conn.execute(statement.format(turn=REPLY_TURN[shape]))
    return change


CHANGES = {
    "annotated": annotate_reply,
    "other-person": sql("UPDATE turn_sources SET contact_id='someone-else' WHERE turn_id='{turn}'"),
    "attribution-invalidated": sql("INSERT INTO source_attribution_invalidations VALUES ('{turn}', '{turn}', 'op')"),
    "session-checkpoint": sql("UPDATE turn_sources SET scope='session' WHERE turn_id='{turn}'"),
}


@pytest.mark.parametrize("shape", ["later-turn", "envelope"])
@pytest.mark.parametrize("change", CHANGES)
def test_an_erased_annotated_or_out_of_scope_reply_is_not_attached(tmp_path, shape, change):
    ledger = ledger_at(tmp_path)
    SHAPES[shape](ledger)
    if shape == "envelope" and change in {"other-person", "attribution-invalidated", "session-checkpoint"}:
        # The question shares the reply's source: out of scope, it is not found either.
        CHANGES[change](ledger, shape)
        assert ledger.search_sources("shrubs", **SCOPE) == []
        return
    hits = question_hits(ledger)
    CHANGES[change](ledger, shape)
    rows = prepare(ledger, hits)
    assert not any(row["kind"] == "conversation_pair" for row in rows)
    assert not any(ANSWER in str(row.get("content")) for row in rows)
    row, = rows
    assert (row["kind"], row["role"], row["content"]) == ("source_quote", "user", ASKED)


# --- 7. Ambiguous replies form no pair ------------------------------------------------------------------------------

@pytest.mark.parametrize("second", ["later-turn", "envelope-and-later-turn"])
def test_ambiguous_replies_form_no_pair(tmp_path, second):
    ledger = ledger_at(tmp_path)
    if second == "later-turn":
        asked(ledger, "input")
        answered(ledger, "reply", parent="input")
        answered(ledger, "regenerated", parent="input", reply="Boxwood, clipped low, would also suit it.")
    else:
        envelope(ledger, "input")
        answered(ledger, "regenerated", parent="input", reply="Boxwood, clipped low, would also suit it.")
    row, = prepare(ledger, question_hits(ledger))
    assert (row["kind"], row["role"], row["content"]) == ("source_quote", "user", ASKED)


def test_a_question_with_no_reply_stays_a_quotation(tmp_path):
    ledger = ledger_at(tmp_path)
    asked(ledger, "input")
    row, = prepare(ledger, question_hits(ledger))
    assert (row["kind"], row["role"], row["content"]) == ("source_quote", "user", ASKED)


@pytest.mark.parametrize("form", ["excerpt", "claims"])
async def test_a_question_recalled_as_excerpts_or_claims_brings_no_loose_reply(tmp_path, form):
    """A reply stands only beside its input as a plain quotation: when the question is found as an excerpt of a long
    message, or its words are recalled as claims, the answer is not shown by itself."""
    ledger = ledger_at(tmp_path)
    question = "ordinary context " * 150 + ASKED if form == "excerpt" else "My north fence is shady."
    ledger.record_source("turn", contact_id="person", session_id="original", derive_claims=form == "claims",
                         messages=[{"role": "user", "content": question}, {"role": "assistant", "content": ANSWER}])
    if form == "claims":
        assert await SourceClaimProjection(ledger).process_one(Model({question: claim(
            question, "shady", subject="north fence", predicate="light")}))
    hits = [hit for hit in ledger.search_sources("shrubs" if form == "excerpt" else "fence", **SCOPE, limit=20)
            if hit["role"] == "user"]
    assert hits and all(hit.get("excerpt_truncated", False) for hit in hits) == (form == "excerpt")
    rows = prepare(ledger, hits)
    assert rows and not any(row["kind"] == "conversation_pair" for row in rows)
    assert not any(ANSWER in str(row.get("content")) for row in rows)


# --- 8. A packet already holding the pair does not hold it twice ----------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
async def test_a_packet_holds_the_pair_once_when_both_messages_are_found(tmp_path, monkeypatch, shape):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = ledger_at(tmp_path)
    SHAPES[shape](ledger)
    both = ledger.search_sources("shrubs hollies", **SCOPE, limit=20)
    assert sorted(hit["role"] for hit in both) == ["assistant", "user"]
    for hits in (both, list(reversed(both))):
        rows = prepare(ledger, hits)
        assert [row["kind"] for row in rows] == ["conversation_pair"]
        result = await packet(ledger, hits)
        assert [row["kind"] for row in result.selected] == ["conversation_pair"]
        assert result.content.count(ASKED) == 1 and result.content.count(ANSWER) == 1


@pytest.mark.parametrize("answers", ["the same", "different"])
def test_the_same_question_and_answer_in_two_conversations_is_one_pair(tmp_path, answers):
    """Recall keeps one of a person's repeated words (selection collapses repeated plain quotations): a question now
    brings its answer, so the same question with the same answer in two conversations is one pair. Another answer
    is other evidence and stays."""
    ledger = ledger_at(tmp_path)
    envelope(ledger, "monday", session="monday")
    envelope(ledger, "friday", session="friday",
             reply=ANSWER if answers == "the same" else "Ferns, since the fence gets no sun.")
    hits = ledger.search_sources("shrubs", **SCOPE, limit=20)
    assert sorted(hit["turn_id"] for hit in hits) == ["friday", "monday"]
    rows = prepare(ledger, hits)
    assert [row["kind"] for row in rows] == ["conversation_pair"] * (1 if answers == "the same" else 2)
    assert rows[0]["source_turn_id"] == hits[0]["turn_id"]


CORRECTION = "Hollies were a guess; the nursery advised ferns."


@pytest.mark.parametrize("corrected", ["first", "second", "both"])
@pytest.mark.parametrize("message", ["question", "answer"])
async def test_a_repeated_pair_is_one_pair_only_when_the_one_kept_brings_every_correction(
        tmp_path, monkeypatch, corrected, message):
    """The same question and answer in two conversations collapse to the first pair only when that pair brings every
    correction the other one would: an owner correction of the second conversation's question (or of its answer, when
    lanes found both) is never dropped with its pair."""
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = ledger_at(tmp_path)
    envelope(ledger, "monday", session="monday")
    envelope(ledger, "friday", session="friday")
    hits = ledger.search_sources("shrubs hollies" if message == "answer" else "shrubs", **SCOPE, limit=20)
    order = list(dict.fromkeys(hit["turn_id"] for hit in hits))
    assert sorted(order) == ["friday", "monday"]
    targets = {"first": order[:1], "second": order[1:], "both": order}[corrected]
    for n, turn in enumerate(targets):
        ref = ledger.source_references([turn], **SCOPE)[0]
        ledger.append_source_annotation(**SCOPE, annotation_id=f"owner-note-{n}", **ref,
                                        excerpt=ASKED if message == "question" else ANSWER,
                                        correction=f"{CORRECTION} ({turn})", author_principal="person")
    rows = prepare(ledger, hits)
    assert {row["kind"] for row in rows} == {"conversation_pair"}
    assert len(rows) == (1 if corrected == "first" else 2)
    result = await packet(ledger, hits)
    for turn in targets:
        assert f"{CORRECTION} ({turn})" in result.content


class _AnswerReranker:
    """Scores a document that holds the answer high and every other one low, and keeps what it was sent."""

    def __init__(self, answer):
        self.answer, self.documents = answer, []

    def calibration_metadata(self):
        return {"model": "neutral-fixture", "format_version": "v1"}

    async def rerank(self, query, documents, top_k):
        self.documents.append(list(documents))
        return [{"index": i, "score": .95 if self.answer in text else .10} for i, text in enumerate(documents)]


@pytest.mark.parametrize("found", ["the question", "the question and, far below, the answer"])
async def test_a_pair_stands_where_its_question_was_found(tmp_path, monkeypatch, found):
    """A pair takes the place of the better ranked of its two messages, with its rank: a question found first with
    twenty other conversations after it is paired in first place, and the reranker's batch of twenty holds it."""
    from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
    from colonymind.memory.selection import RERANK_SUBMISSION_LIMIT
    ledger = ledger_at(tmp_path)
    envelope(ledger, "fence", session="fence")
    for n in range(RERANK_SUBMISSION_LIMIT):
        ledger.record_source(f"bulbs-{n}", contact_id="person", session_id=f"bulbs-{n}", derive_claims=False,
                             messages=[{"role": "user", "content": f"Which bulbs suit border {n} by the path?"}])
    others = ledger.search_sources("bulbs", **SCOPE, limit=RERANK_SUBMISSION_LIMIT)
    assert len({hit["session_id"] for hit in others}) == RERANK_SUBMISSION_LIMIT
    hits = question_hits(ledger) + others
    if found != "the question":
        hits += [hit for hit in ledger.search_sources("hollies", **SCOPE, limit=20) if hit["role"] == "assistant"]
    rows = prepare(ledger, hits)
    assert len(rows) == RERANK_SUBMISSION_LIMIT + 1
    assert rows[0]["kind"] == "conversation_pair" and rows[0]["relevance"] == 1 / 61
    assert [row["content"] for row in rows[1:]] == [hit["content"] for hit in others]

    reranker = _AnswerReranker("Hollies")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", ".5")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", calibration_fingerprint(
        provider_calibration_metadata(reranker)))
    selector = RecallSelector(reranker.rerank, calibration_metadata=lambda: provider_calibration_metadata(reranker))
    result = await select_memory(CollectedSources(ledger, **SCOPE, watermark=ledger.erasure_watermark("person"),
                                                  hits=list(hits)),
                                 query="Which shrubs did you recommend for the fence?", selector=selector,
                                 limit=None, timezone_name="UTC")
    submitted, = reranker.documents
    assert len(submitted) == RERANK_SUBMISSION_LIMIT and any(ANSWER in document for document in submitted)
    assert [row["kind"] for row in result.selected] == ["conversation_pair"]
    assert ASKED in result.content and ANSWER in result.content


# --- 9. A later reply is found through the map, with no scan of the sources ----------------------------------------

def test_a_later_reply_is_found_through_the_map_without_scanning_the_sources(tmp_path, monkeypatch):
    from colonymind.turns import source_replies
    ledger = ledger_at(tmp_path)
    for n in range(30):
        envelope(ledger, f"other-{n}", question=f"Unrelated question {n}?", reply=f"Unrelated answer {n}.")
    asked(ledger, "input")
    answered(ledger, "reply", parent="input")
    hits = question_hits(ledger)
    statements, connect = [], ledger._connect

    def traced():
        conn = connect()
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(ledger, "_connect", traced)
    row, = prepare(ledger, hits)
    monkeypatch.setattr(ledger, "_connect", connect)
    assert row["kind"] == "conversation_pair"
    assert any(source_replies.TABLE in statement for statement in statements)
    with closing(connect()) as conn:
        plans = {statement: " | ".join(step[3] for step in conn.execute("EXPLAIN QUERY PLAN " + statement))
                 for statement in statements if statement.lstrip().upper().startswith(("SELECT", "WITH"))}
    scans = {statement: plan for statement, plan in plans.items()
             if "SCAN turn_sources" in plan or "SCAN s " in plan + " " or plan.endswith("SCAN s")}
    assert scans == {}
    lookup, = [plan for statement, plan in plans.items() if source_replies.TABLE in statement]
    assert "USING INDEX" in lookup or "USING COVERING INDEX" in lookup


# --- The map: rebuilt the same, built on upgrade ----------------------------------------------------------------------

def test_a_rebuild_is_idempotent_and_matches_what_recording_wrote(tmp_path):
    ledger = ledger_at(tmp_path)
    envelope(ledger, "turn")
    asked(ledger, "input")
    answered(ledger, "reply", parent="input")
    ledger.record_source("checkpoint", contact_id="person", session_id="original", scope="session",
                         derive_claims=False, messages=[USER, {"role": "assistant", "content": ANSWER}])
    written = reply_rows(ledger)
    assert len(written) == 2                                      # a checkpoint's copies pair nothing
    assert ledger.rebuild_source_replies() == 4
    assert reply_rows(ledger) == written
    assert ledger.rebuild_source_replies() == 4 and reply_rows(ledger) == written


def downgrade(path):
    """The store as 1.11 left it: no reply map, no triggers, no queue."""
    from colonymind.turns import source_replies
    with closing(sqlite3.connect(path)) as conn, conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE ?",
                                    (source_replies.TABLE + "%",)).fetchall():
            conn.execute(f'DROP TRIGGER "{name}"')
        conn.execute(f"DROP TABLE {source_replies.TABLE}")
        conn.execute(f"DROP TABLE {source_replies.PENDING}")


def test_the_upgrade_builds_the_reply_map_for_an_existing_store(tmp_path, caplog):
    ledger = ledger_at(tmp_path)
    envelope(ledger, "turn")
    asked(ledger, "input")
    answered(ledger, "reply", parent="input")
    written = reply_rows(ledger)
    downgrade(tmp_path / "turn-idempotency.db")
    with caplog.at_level("INFO", logger="colonymind.turns.idempotency"):
        upgraded = ledger_at(tmp_path)
    assert reply_rows(upgraded) == written
    assert any("reply map" in record.getMessage() and "3 sources" in record.getMessage()
               for record in caplog.records)


def test_sources_written_by_an_older_version_are_mapped_when_the_ledger_opens(tmp_path):
    ledger = ledger_at(tmp_path)
    messages = [USER, {"role": "assistant", "content": ANSWER}]
    with closing(sqlite3.connect(ledger.db_path)) as conn, conn:
        conn.execute("INSERT INTO turn_sources(turn_id, content_sha256, contact_id, session_id, scope, messages_json)"
                     " VALUES ('older', ?, 'person', 'original', 'person', ?)", ("0" * 64, json.dumps(messages)))
    assert reply_rows(ledger) == []
    assert len(reply_rows(ledger_at(tmp_path))) == 1


# --- What the person sees: context assembly shows the answer beside the question ------------------------------------

@pytest.fixture
def source_app(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(host.v2_router)
    return app


async def test_context_assembly_shows_the_answer_to_a_recalled_question(source_app):
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url="http://test") as client:
        response = await client.put("/v2/host/turns/turn-said", json={
            "identity": {"host_id": "test-host"},
            "context": {"session_id": "session-a", "contact_id": "contact-a", "channel_id": "test:thread-a",
                        "turn_id": "turn-said"},
            "user_message": {"role": "user", "content": ASKED},
            "assistant_message": {"role": "assistant", "content": ANSWER}})
        assert response.status_code == 201, response.text
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test-host"},
            "context": {"contact_id": "contact-a", "session_id": "session-b"},
            "incoming_message": {"role": "user", "content": "What did you say about shrubs?"}})
        assert response.status_code == 200, response.text
        memory = "\n".join(section["body"] for section in response.json()["sections"]
                           if section["id"] == "colonymind-memory")
        assert ASKED in memory and ANSWER in memory and "conversation_pair" in memory


def test_forgetting_the_reply_message_withholds_the_recalled_reply(tmp_path):
    from colonymind.memory import curated
    ledger = ledger_at(tmp_path)
    envelope(ledger, "turn")
    assert len(reply_rows(ledger)) == 1
    ledger.record_source('copy', contact_id='person', session_id='original',
                         messages=[{'role': 'assistant', 'content': ANSWER}], derive_claims=False)
    curated.forget_sources(ledger, 'person', source_ids=['copy'])
    rows = ledger.search_sources(ANSWER, **SCOPE)
    assert not curated.current_candidates(ledger, rows, contact_id='person')
    assert len(reply_rows(ledger)) == 1  # historical reply index is retained

from colonymind.memory import curated

@pytest.mark.parametrize("erased", ["input", "reply"])
def test_erasing_either_source_removes_its_reply_rows(tmp_path, erased):
    ledger = ledger_at(tmp_path)
    asked(ledger, "input")
    answered(ledger, "reply", parent="input")
    envelope(ledger, "kept", question="Is the gate fixed?", reply="Unrelated, and kept.")
    assert {row[0] for row in reply_rows(ledger)} == {"input", "kept"}
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'person', source_ids=[erased])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
