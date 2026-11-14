"""The keyword lanes: rarity-chosen terms, and a stemmed lane with CJK bigrams (docs/RECALL-HYBRID.md, Keyword lanes).

The keyword query kept the first 12 words of a question, so a long message searched only its opening; the full-text
table did not stem, so "moved" did not find "moving", and an unspaced CJK run was one token, so "东京" did not find
"我昨天去了东京出差". The query now keeps its identifiers and then the rarest words; a second full-text table indexes
the same chunks stemmed, with every CJK run rewritten as overlapping two-character pieces, and is a lane of its own in
``collect_sources``.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import sqlite3
import threading
import time
from contextlib import closing

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.errors import install_exception_handlers
from colonymind.api.routers import host
from colonymind.memory import search, trace as recall_trace
from colonymind.memory.search import collect_sources
from colonymind.turns import idempotency, keywords
from colonymind.turns.idempotency import TurnIdempotencyLedger, source_message_hash
from colonymind.turns.source_vectors import SourceVectors
from colonymind.util.temporal import strip_arrival_stamps

SCOPE = {"contact_id": "owner", "session_id": "later"}


def ledger_at(tmp_path, name="turn-idempotency.db"):
    return TurnIdempotencyLedger(tmp_path / name)


def record(ledger, turn_id, *contents, session="s-1", scope="person", contact="owner", role="user"):
    ledger.record_source(turn_id, contact_id=contact, session_id=session, scope=scope,
                         messages=[{"role": role, "content": content} for content in contents],
                         derive_claims=False)


def stem_rows(ledger):
    """The stemmed index: each row's source, speaker, chunk index and indexed text."""
    with closing(ledger._connect()) as conn:
        return sorted(tuple(row) for row in conn.execute(
            "SELECT m.turn_id, m.role, m.chunk, f.content FROM turn_source_stem_chunks m "
            "JOIN turn_source_stems f ON f.rowid = m.id"))


def stem_count(ledger):
    with closing(ledger._connect()) as conn:
        return (conn.execute("SELECT count(*) FROM turn_source_stems").fetchone()[0],
                conn.execute("SELECT count(*) FROM turn_source_stem_chunks").fetchone()[0])


def pending(ledger):
    with closing(ledger._connect()) as conn:
        return sorted(row[0] for row in conn.execute("SELECT turn_id FROM turn_source_stem_pending"))


def expression_before(query):
    """The 1.11 keyword query, verbatim: identifiers first, then the first words, 12 in all."""
    stop = {"the", "and", "that", "this", "what", "when", "where", "how", "you", "your", "was", "were", "are", "for",
            "with", "remember", "about"}
    text = strip_arrival_stamps(query[:4096])
    identifiers = list(dict.fromkeys(
        piece.lower() for piece in re.findall(r"[^\W_]+(?:[-./][^\W_]+)+", text)
        if any(len(part) <= 2 for part in re.findall(r"\w+", piece))))[:4]
    words = list(dict.fromkeys(
        [*identifiers, *(word.lower() for word in re.findall(r"\w+", text)
                         if len(word) > 2 and word.lower() not in stop)]))[:12]
    if not words:
        return None
    return " OR ".join('"' + word.replace('"', '') + '"' for word in words)


def lexical_expression(ledger, query):
    with closing(ledger._connect()) as conn:
        return keywords.match_expression(keywords.query_terms(conn, query))


COMMON = ("please check whether our weekly planning notes mention anything important regarding tomorrow morning "
          "meeting schedule because everyone keeps asking again").split()


# --- 1. A discriminative term late in a long message finds its source ------------------------------------------------

def test_a_discriminative_term_at_word_20_finds_its_source(tmp_path):
    ledger = ledger_at(tmp_path)
    # Many sources share the question's opening words; one holds its only rare word.
    for index in range(30):
        record(ledger, f"common-{index}", " ".join(COMMON) + f" note {index}.")
    record(ledger, "answer", "The zeppelin hangar code is 7741.")
    question = " ".join(COMMON[:19] + ["zeppelin"])
    assert question.split().index("zeppelin") == 19                       # the 20th word
    assert "zeppelin" not in expression_before(question)                    # 1.11 never searched it
    hits = ledger.search_sources(question, **SCOPE, limit=10)
    assert "answer" in [hit["turn_id"] for hit in hits]
    assert '"zeppelin"' in lexical_expression(ledger, question)
    # The stemmed lane chooses its terms the same way.
    assert "answer" in [hit["turn_id"] for hit in ledger.search_stems(question, **SCOPE, limit=10)]


def test_identifiers_stay_first_and_the_rest_are_the_rarest(tmp_path):
    ledger = ledger_at(tmp_path)
    for index in range(12):
        record(ledger, f"common-{index}", " ".join(COMMON))
    record(ledger, "rare", "a lighthouse keeper and a zeppelin pilot")
    question = "Order p-72 and B-12: " + " ".join(COMMON) + " lighthouse zeppelin"
    with closing(ledger._connect()) as conn:
        terms = keywords.query_terms(conn, question)
    assert len(terms) == keywords.MAX_TERMS == 12
    assert terms[:2] == ["p-72", "b-12"]
    assert {"lighthouse", "zeppelin"} <= set(terms)
    # The chosen words keep the question's order.
    words = [term for term in terms if term not in ("p-72", "b-12")]
    order = [word.lower() for word in re.findall(r"\w+", question)]
    assert words == sorted(words, key=order.index)


def test_a_word_no_chunk_holds_never_takes_the_place_of_one_that_does(tmp_path):
    """Rarest means rarest among the words the index holds: a word no chunk holds can find nothing, so it never
    takes a place from one some chunk holds. A long question whose instructions name no stored word (a format to
    write, a field list) still searches the words its sources hold, wherever they stand in it."""
    ledger = ledger_at(tmp_path)
    for index in range(3):
        record(ledger, f"common-{index}", "The design brief pickup is on the planning board.")
    record(ledger, "question", "A question about two recorded statements that disagree.")
    question = ("When is the design brief pickup with p-75? Write answer.json as exactly {\"status\": string, "
                "\"time\": string, \"question\": string}: status \"answer\" with the time as HH:MM if what I told "
                "you is consistent, or status \"ask\" with an empty time.")
    with closing(ledger._connect()) as conn:
        held = keywords.document_counts(conn, ["design", "brief", "pickup", "question"])
        terms = keywords.query_terms(conn, question)
        stemmed = keywords.query_terms(conn, question, stemmed=True)
    assert held == {"design": 3, "brief": 3, "pickup": 3, "question": 1}
    assert len(terms) == keywords.MAX_TERMS and terms[0] == "p-75"
    for chosen in (terms, stemmed):
        assert {"design", "brief", "pickup", "question"} <= set(chosen)
    # The words are still read in the question's order.
    order = [word.lower() for word in re.findall(r"\w+", question)]
    assert terms[1:] == sorted(terms[1:], key=order.index)
    # The source that holds only the late word is found by both lanes.
    assert "question" in [hit["turn_id"] for hit in ledger.search_sources(question, **SCOPE, limit=10)]
    assert "question" in [hit["turn_id"] for hit in ledger.search_stems(question, **SCOPE, limit=10)]


# --- 2. A query of 12 terms or fewer produces the identical expression ----------------------------------------------

@pytest.mark.parametrize("question", [
    "Where did I put the blue tin with the spare key?",
    "What is order p-72 and the 4.2m cable, answer.json?",
    "remember the boiler service Friday",
    "我昨天去了东京出差 and the hotel",
    "Café crème résumé déjà vu",
    "[2026-09-28 10:00] What did Sam say about the parcel depot?",
    "the and that",
    "",
    "one two three four five six seven eight nine ten eleven twelve",
])
def test_a_query_of_twelve_terms_or_fewer_builds_the_same_expression(tmp_path, monkeypatch, question):
    ledger = ledger_at(tmp_path)
    record(ledger, "any", "Some words to count.")

    def unread(*args, **kwargs):
        raise AssertionError("a query of 12 terms or fewer reads no statistics")
    monkeypatch.setattr(keywords, "document_counts", unread)
    assert lexical_expression(ledger, question) == expression_before(question)


# --- 3. A failed statistics read falls back to position order --------------------------------------------------------

def test_a_failed_statistics_read_falls_back_to_position_order(tmp_path, monkeypatch):
    ledger = ledger_at(tmp_path)
    record(ledger, "answer", "The zeppelin hangar code is 7741.")
    question = " ".join(COMMON[:19] + ["zeppelin"])

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("no such module: fts5vocab")
    monkeypatch.setattr(keywords, "document_counts", broken)
    assert lexical_expression(ledger, question) == expression_before(question)
    assert ledger.search_sources(question, **SCOPE) == []                   # searched, by the first 12 words
    with closing(ledger._connect()) as conn:                               # the stemmed lane falls back too
        assert keywords.query_terms(conn, question, stemmed=True) == question.split()[:12]


def test_the_statistics_read_counts_documents_per_term(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "a", "zeppelin lighthouse lighthouse")
    record(ledger, "b", "lighthouse keeper")
    record(ledger, "c", "我昨天去了东京出差")
    with closing(ledger._connect()) as conn:
        assert keywords.document_counts(conn, ["lighthouse", "zeppelin", "absent"]) == {
            "lighthouse": 2, "zeppelin": 1}
        # Each index counts what its tokenizer makes of a term: folded, and for the stemmed lane stemmed and with
        # CJK pieces as tokens.
        assert keywords.document_counts(conn, ["LIGHTHOUSE", "Zeppélin"]) == {"LIGHTHOUSE": 2, "Zeppélin": 1}
        assert keywords.document_counts(conn, ["东京", "lighthouses", "keepers"], stemmed=True) == {
            "东京": 1, "lighthouses": 2, "keepers": 1}
        assert keywords.document_counts(conn, ["lighthouses", "keepers"]) == {}


# --- 4. The keyword lanes stay fast on a large store -----------------------------------------------------------------

def test_the_keyword_lanes_stay_under_50_ms_at_p90_on_50000_messages(tmp_path):
    """The bound, 50 ms at the 90th percentile over 30 questions, is the unit's own choice (not a measured rule), and so
    is the fixture: 400 made-up stems in eight inflections each, their frequencies shuffled, drawn by a Zipf curve
    shifted so that no word is in more than about one message in eight, as with chat text once stop words and short
    words are dropped."""
    rng = random.Random(20260928)
    stems = sorted({a + b + c + d + e for a in "bdfgklmnprstvz" for b in "aeiou" for c in "bdfgklmnprstvz"
                    for d in "aeiou" for e in "dlmnrst"})
    vocabulary = [stem + suffix for stem in rng.sample(stems, 400)
                  for suffix in ("", "s", "ing", "ed", "er", "ly", "ment", "ness")]
    rng.shuffle(vocabulary)
    weights = [1 / (rank + 60) for rank in range(len(vocabulary))]
    ledger = ledger_at(tmp_path)
    started = time.perf_counter()
    with closing(ledger._connect()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        for index in range(50_000):
            messages = [{"role": "user" if index % 2 else "assistant",
                         "content": " ".join(rng.choices(vocabulary, weights, k=rng.randint(10, 60)))}]
            conn.execute("INSERT INTO turn_sources(turn_id, content_sha256, contact_id, session_id, scope, "
                         "messages_json, ingested_at) VALUES (?, ?, 'owner', 's-1', 'person', ?, ?)",
                         (f"t-{index}", f"{index:064x}", json.dumps(messages), "2026-09-28T00:00:00Z"))
            TurnIdempotencyLedger._index_messages(conn, f"t-{index}", messages)
    assert time.perf_counter() - started < 120
    assert not pending(ledger)
    questions = [" ".join(rng.choices(vocabulary, weights, k=rng.randint(4, 80))) for _ in range(30)]
    for lane in (ledger.search_sources, ledger.search_stems):
        lane(questions[0], **SCOPE, limit=10)                               # the first open warms the cache
        timings = []
        for question in questions:
            begun = time.perf_counter()
            assert lane(question, **SCOPE, limit=10)
            timings.append((time.perf_counter() - begun) * 1000)
        timings.sort()
        assert timings[int(0.9 * len(timings)) - 1] < 50, (lane.__name__, timings)


def test_ranking_the_best_matches_first_gives_the_single_querys_results(tmp_path, monkeypatch):
    """The lanes rank their best full-text matches before reading any source; when those fall short (here, other
    people's stronger matches fill the window) the lane reads on, and the results are what one scope-checked query
    ordered by bm25 gives."""
    ledger = ledger_at(tmp_path)
    for index in range(12):
        record(ledger, f"sam-{index}", "harbour harbour harbour ferry " * 3, contact="sam")
    for index in range(8):
        record(ledger, f"mine-{index}", f"The harbour ferry note {index} about moving boxes " + "filler " * index)
    record(ledger, "private", "harbour moving", session="s-9", scope="session")
    wide = {lane: getattr(ledger, lane)("harbour ferry moving", **SCOPE, limit=20)
            for lane in ("search_sources", "search_stems")}
    assert {hit["turn_id"] for hit in wide["search_sources"]} == {f"mine-{index}" for index in range(8)}
    monkeypatch.setattr(idempotency, "MATCH_WINDOW", 4)
    for lane, hits in wide.items():
        assert getattr(ledger, lane)("harbour ferry moving", **SCOPE, limit=20) == hits
        assert getattr(ledger, lane)("harbour ferry moving", **SCOPE, limit=3) == hits[:3]
    with closing(ledger._connect()) as conn:
        expression = keywords.match_expression(keywords.query_terms(conn, "harbour ferry moving"))
        single = [row["turn_id"] for row in conn.execute(
            "SELECT f.turn_id FROM turn_source_search AS f JOIN turn_sources AS s ON s.turn_id=f.turn_id "
            "WHERE turn_source_search MATCH ? AND s.contact_id='owner' AND (s.scope='person' OR s.session_id='later')"
            " ORDER BY bm25(turn_source_search), f.rowid", (expression,))]
    assert [hit["turn_id"] for hit in wide["search_sources"]] == single


# --- 5. and 6. Inflections and CJK words find their sources ---------------------------------------------------------

def test_moved_finds_moving_through_the_stemmed_lane(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "move", "I was moving house all last week.")
    assert ledger.search_sources("when I moved", **SCOPE) == []             # the unstemmed lane misses it
    hits = ledger.search_stems("when I moved", **SCOPE)
    assert [hit["turn_id"] for hit in hits] == ["move"]
    assert hits[0]["content"] == "I was moving house all last week."


@pytest.mark.parametrize("stored, question", [
    ("我昨天去了东京出差", "东京"),
    ("我昨天去了东京出差", "我去过东京吗"),
    ("昨日は東京タワーに行きました", "東京タワー"),
    ("주말에 서울에서 만났어요", "서울"),
])
def test_cjk_words_find_their_sources(tmp_path, stored, question):
    ledger = ledger_at(tmp_path)
    record(ledger, "cjk", stored)
    record(ledger, "other", "An unrelated note about the harbour.")
    assert ledger.search_sources(question, **SCOPE) == []                   # one token under unicode61
    hits = ledger.search_stems(question, **SCOPE)
    assert [hit["turn_id"] for hit in hits] == ["cjk"] and hits[0]["content"] == stored


def test_the_cjk_rewrite_is_overlapping_pieces_of_two():
    assert keywords.cjk_bigrams("我昨天去了东京出差") == " 我昨 昨天 天去 去了 了东 东京 京出 出差 "
    assert keywords.cjk_bigrams("東京タワー!") == " 東京 京タ タワ ワー !"
    assert keywords.cjk_bigrams("猫") == " 猫 "
    assert keywords.cjk_bigrams("Tokyo東京trip。서울") == "Tokyo 東京 trip。 서울 "
    assert keywords.cjk_bigrams("plain English, unchanged") == "plain English, unchanged"


# --- 7. Erasure removes rows from the new table ---------------------------------------------------------------------


def test_any_writer_that_deletes_or_rewrites_a_source_leaves_no_stemmed_rows(tmp_path):
    """The rows follow the canonical source whoever changes it: a view's own erasure deletes its source rows, the
    secrets migration rewrites one under a new id."""
    ledger = ledger_at(tmp_path)
    record(ledger, "entry", "The archive migration plan changed.")
    record(ledger, "raw", "The alarm code is swordfish.")
    with closing(ledger._connect()) as conn, conn:
        conn.execute("DELETE FROM turn_source_search WHERE turn_id='entry'")
        conn.execute("DELETE FROM turn_sources WHERE turn_id='entry'")
    assert ledger.search_stems("archive migration", **SCOPE) == []
    sealed = [{"role": "user", "content": "The alarm code is [sealed]."}]
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE turn_sources SET turn_id='resealed', messages_json=? WHERE turn_id='raw'",
                     (json.dumps(sealed),))
        conn.execute("DELETE FROM turn_source_search WHERE turn_id='raw'")
        TurnIdempotencyLedger._index_messages(conn, "resealed", sealed)
    assert ledger.search_stems("swordfish", **SCOPE) == []
    assert [(row[0], row[3]) for row in stem_rows(ledger)] == [("resealed", "The alarm code is [sealed].")]
    assert not pending(ledger)


# --- 8. A rebuild is idempotent -------------------------------------------------------------------------------------

def test_a_rebuild_is_idempotent_and_matches_what_ingestion_wrote(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "a", "I was moving house.", "我昨天去了东京出差")
    record(ledger, "b", "x" * 1900 + " harbour " + "y" * 2500, role="assistant")
    record(ledger, "c", "The bakery opens early.", session="s-2", scope="session")
    written = stem_rows(ledger)
    assert len(written) == 6                                               # "a" is two chunks, "b" three
    assert ledger.rebuild_keyword_stems() == 3
    assert stem_rows(ledger) == written
    assert ledger.rebuild_keyword_stems() == 3
    assert stem_rows(ledger) == written and stem_count(ledger) == (6, 6) and not pending(ledger)


# --- 9. The upgrade builds the table on a store without it ----------------------------------------------------------

def downgrade(path):
    """The store as 1.11 left it: no stemmed index, no triggers, no queue."""
    with closing(sqlite3.connect(path)) as conn, conn:
        for name, kind in conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('trigger', 'table') "
                                       "AND name LIKE 'turn_source_stem%'").fetchall():
            if kind == "trigger":
                conn.execute(f'DROP TRIGGER "{name}"')
        conn.execute("DROP TABLE turn_source_stems")
        conn.execute("DROP TABLE turn_source_stem_chunks")
        conn.execute("DROP TABLE turn_source_stem_pending")


def test_the_upgrade_builds_the_stemmed_index_for_an_existing_store(tmp_path, caplog):
    first = ledger_at(tmp_path)
    record(first, "a", "I was moving house.")
    record(first, "b", "我昨天去了东京出差", session="s-2", scope="session")
    written = stem_rows(first)
    downgrade(tmp_path / "turn-idempotency.db")
    with caplog.at_level("INFO", logger="colonymind.turns.idempotency"):
        upgraded = ledger_at(tmp_path)
    assert stem_rows(upgraded) == written and not pending(upgraded)
    assert [hit["turn_id"] for hit in upgraded.search_stems("moved", **SCOPE)] == ["a"]
    assert any("stemmed keyword index" in record.getMessage() and "2 sources" in record.getMessage()
               for record in caplog.records)


def test_an_interrupted_build_resumes_and_sources_written_by_an_older_version_are_indexed(tmp_path):
    """Every new or rewritten source is queued in SQL, so a build cut short, or sources written while an older version
    ran (a rollback), are indexed the next time the ledger opens."""
    ledger = ledger_at(tmp_path)
    record(ledger, "a", "I was moving house.")
    older = [{"role": "user", "content": "The ferries were cancelled."}]
    with closing(sqlite3.connect(tmp_path / "turn-idempotency.db")) as conn, conn:
        # What 1.11 writes: the source and its unstemmed rows, nothing stemmed.
        conn.execute("INSERT INTO turn_sources(turn_id, content_sha256, contact_id, session_id, scope, messages_json)"
                     " VALUES ('older', ?, 'owner', 's-1', 'person', ?)", ("0" * 64, json.dumps(older)))
        conn.execute("INSERT INTO turn_source_search(turn_id, role, content) VALUES ('older', 'user', ?)",
                     (older[0]["content"],))
    assert pending(ledger) == ["older"] and ledger.search_stems("cancelling", **SCOPE) == []
    reopened = ledger_at(tmp_path)
    assert pending(reopened) == []
    assert [hit["turn_id"] for hit in reopened.search_stems("cancelling ferry", **SCOPE)] == ["older"]


# --- 10. A hit resolves to its canonical chunk and message hash -----------------------------------------------------

def test_a_stemmed_hit_resolves_to_its_canonical_chunk_and_message_hash(tmp_path):
    ledger = ledger_at(tmp_path)
    long = "ordinary context " * 180 + "The hydrofoils departed Friday at nine."
    record(ledger, "long", "Short opener.", long, session="s-1")
    hits = ledger.search_stems("hydrofoil departure", contact_id="owner", session_id="s-1")
    assert hits and all(hit["turn_id"] == "long" for hit in hits)
    hit = hits[0]
    chunks = [long[start:start + 2000] for start in range(0, len(long), 1800)]
    assert hit["content"] in chunks and "hydrofoils departed" in hit["content"]
    assert hit["role"] == "user" and hit["excerpt_truncated"] is True
    assert hit["source_message_hash"] == source_message_hash("s-1", {"role": "user", "content": long})
    # The same excerpt the unstemmed lane returns for it: one candidate, not two.
    plain = ledger.search_sources("hydrofoils", contact_id="owner", session_id="s-1")
    assert {(row["turn_id"], row["role"], row["content"], row["source_message_hash"]) for row in plain} == {
        (row["turn_id"], row["role"], row["content"], row["source_message_hash"]) for row in hits}


def test_a_stale_stemmed_row_never_spends_the_result_budget(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "a", "The ferries were cancelled.")
    with closing(ledger._connect()) as conn, conn:
        # An indexed text that is not the rewrite of the canonical chunk at its place (it cannot happen through the
        # writers; the check is what keeps it out if it ever did).
        conn.execute("UPDATE turn_source_stems SET content='cancelled lighthouse' "
                     "WHERE rowid=(SELECT id FROM turn_source_stem_chunks WHERE turn_id='a')")
    assert ledger.search_stems("lighthouse", **SCOPE) == []


def test_the_stemmed_lane_keeps_the_scope(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "mine", "I was moving house.")
    record(ledger, "theirs", "I was moving house too.", contact="sam")
    record(ledger, "private", "Moving the piano tomorrow.", session="s-9", scope="session")
    assert [hit["turn_id"] for hit in ledger.search_stems("moved", **SCOPE)] == ["mine"]
    assert {hit["turn_id"] for hit in ledger.search_stems("moved", contact_id="owner", session_id="s-9")} == {
        "mine", "private"}


# --- The lane registry: every lane fused in collect_sources, and traced ---------------------------------------------

async def test_the_stemmed_lane_is_fused_and_traced_in_collect_sources(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "move", "I was moving house all last week.")
    record(ledger, "both", "Moved the piano; moving again soon.")
    assert [lane.name for lane in search.LANES] == ["lexical", "vector", "stemmed", "claim_key", "window"]
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(ledger, query="the piano moved", **SCOPE, trace=tracer)
    lexical = ledger.search_sources("the piano moved", **SCOPE, limit=10)
    stemmed = ledger.search_stems("the piano moved", **SCOPE, limit=10)
    assert [hit["turn_id"] for hit in lexical] == ["both"]
    assert {hit["turn_id"] for hit in stemmed} == {"both", "move"}
    # One reciprocal-rank fusion over every lane; the keyword lanes' fusion is the lexical evidence.
    assert collected.hits == search.fuse([lexical, [], stemmed, []])
    assert collected.lexical_hits == search.fuse([lexical, stemmed, []])
    assert [hit["turn_id"] for hit in collected.hits] == ["both", "move"]
    assert collected.semantic == "unavailable"
    assert tracer._lane_hits == {"lexical": 1, "vector": 0, "vector_media": 0, "stemmed": 2, "claim_key": 0}


def test_fusion_is_reciprocal_rank_over_the_lanes_in_order():
    one = {"turn_id": "a", "role": "user", "content": "x", "source_message_hash": "h1"}
    two = {"turn_id": "b", "role": "user", "content": "y", "source_message_hash": "h2"}
    three = {"turn_id": "c", "role": "user", "content": "z", "source_message_hash": "h3"}
    assert search.fuse([[one, two], [two, three]]) == [two, one, three]
    assert search.fuse([[one], [dict(one, retrieval_method="semantic")]]) == [dict(one, retrieval_method="semantic")]
    assert search.fuse([]) == [] and search.fuse([[one, two]]) == [one, two]


async def test_a_keyword_lane_failure_fails_the_collection_and_stops_the_other_lanes(tmp_path, monkeypatch):
    ledger = ledger_at(tmp_path)
    record(ledger, "a", "I was moving house.")
    # The vector lane is still waiting on its index when the stemmed lane fails: the collection fails at once and
    # cancels it, rather than waiting on a lane registered before the failed one.
    vector_started = threading.Event()
    vector_cancelled = []

    async def pending(self, query, **kwargs):
        vector_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            vector_cancelled.append(query)
            raise

    def broken(*args, **kwargs):
        assert vector_started.wait(5)
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(SourceVectors, "search", pending)
    monkeypatch.setattr(ledger, "search_stems", broken)
    # 5 s bounds a hang (my choice); the collection fails in milliseconds.
    with pytest.raises(sqlite3.OperationalError):
        await asyncio.wait_for(collect_sources(ledger, query="moving", **SCOPE), timeout=5)
    assert vector_cancelled == ["moving"]


# --- What the person sees: context assembly recalls inflected and CJK words -----------------------------------------

@pytest.fixture
def source_app(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(host.v2_router)
    return app


@pytest.mark.parametrize("said, asked", [
    ("We are moving the workshop to the harbour next month.", "When did we say we would move?"),
    ("我昨天去了东京出差", "我去过东京吗"),
])
async def test_context_assembly_recalls_what_only_the_stemmed_lane_finds(source_app, said, asked):
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url="http://test") as client:
        response = await client.put("/v2/host/turns/turn-said", json={
            "identity": {"host_id": "test-host"},
            "context": {"session_id": "session-a", "contact_id": "contact-a", "channel_id": "test:thread-a",
                        "turn_id": "turn-said"},
            "user_message": {"role": "user", "content": said},
            "assistant_message": {"role": "assistant", "content": "Noted."}})
        assert response.status_code == 201, response.text
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test-host"},
            "context": {"contact_id": "contact-a", "session_id": "session-b"},
            "incoming_message": {"role": "user", "content": asked}})
        assert response.status_code == 200, response.text
        memory = "\n".join(section["body"] for section in response.json()["sections"]
                           if section["id"] == "colonymind-memory")
        assert said in memory

from colonymind.memory import curated

def test_erasing_a_source_removes_its_stemmed_rows(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "secret", "The locker combination is 4417 near the gardens.")
    record(ledger, "kept", "The gardens open at nine.")
    assert ledger.search_stems("locker combination", **SCOPE)
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'owner', source_ids=['secret'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert ledger.search_stems("locker combination", **SCOPE) == []


def test_erasing_a_message_removes_its_copies_from_the_stemmed_index(tmp_path):
    """A checkpoint keeps other messages: the erased one's copy leaves the stemmed index, the rest stay found."""
    ledger = ledger_at(tmp_path)
    erased, kept = ('My bank pin is 4417 for the locker.', 'The bakery opens early on Sundays.')
    record(ledger, 'turn', erased, session='s-1')
    record(ledger, 'checkpoint', erased, kept, session='s-1', scope='session')
    assert {hit['turn_id'] for hit in ledger.search_stems('locker', contact_id='owner', session_id='s-1')} == {'turn', 'checkpoint'}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['turn'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert ledger.search_stems('locker', contact_id='owner', session_id='s-1') == []
    assert ('checkpoint', keywords.cjk_bigrams(kept)) in [(row[0], row[3]) for row in stem_rows(ledger)]
    assert len(stem_rows(ledger)) == 3
    hit, = ledger.search_stems('bakeries', contact_id='owner', session_id='s-1')
    assert hit['content'] == kept and hit['source_message_hash'] == source_message_hash('s-1', {'role': 'user', 'content': kept})
    assert not pending(ledger)
