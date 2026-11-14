"""A checkpoint's copy of a message never stands in for the message itself.

Before compressing a conversation the host records its user and assistant messages as a checkpoint: a
session-scoped source of its own, visible only in that session. Every message the checkpoint copies from an
attributed turn carries the same source message hash as the original (the hash covers the session, the role and
the words). The copies were never paired with their input, so the reply rule (an agent's reply is shown only beside
the message it answers) never reached them, and the plain-quotation dedupe covers person scope only: in the
conversation itself, the agent's restatement came back without the person's words, once per checkpoint.

A session copy of a message that a person-scope candidate already carries is dropped, so the reply rule acts on
the original. What only a checkpoint holds stays recallable, once.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.memory.recall import (
    RERANK_DOCUMENT_CHARS, calibration_fingerprint, provider_calibration_metadata, source_candidates,
)
from colonymind.memory.search import CollectedSources, collect_sources, select_memory
from colonymind.memory.selection import RecallSelector
from colonymind.turns import TurnIdempotencyLedger, source_annotations
from colonymind.turns.idempotency import source_message_hash
from test_recall_owner_first import NOW, PROBE, REPLY, SAID, _Claims

SESSION = "s-1"
ONLY_IN_CHECKPOINT = "The proof of the quarterly review poster goes to the printer on Thursday."


def checkpoint(ledger, messages, *, occurred_at="2026-09-25T13:00:00+00:00"):
    """What the host's pre-compress hook records: the session's messages, under a content-derived turn id."""
    turn = "checkpoint:" + hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    ledger.record_source(turn, contact_id="owner", session_id=SESSION, scope="session",
                         occurred_at=occurred_at, messages=messages)
    return turn


def correct(ledger, source_id, excerpt, correction, *, annotation_id="correction-1"):
    """The owner's attributed correction of words in one recalled source."""
    ref, = ledger.source_references([source_id], contact_id="owner", session_id=SESSION)
    return source_annotations.append(ledger, contact_id="owner", session_id=SESSION, annotation_id=annotation_id,
                                     source_id=source_id, source_version=ref["source_version"], excerpt=excerpt,
                                     correction=correction, author_principal="owner")


async def recall(ledger, query=PROBE):
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION)
    return await select_memory(collected, query=query, selector=RecallSelector(), timezone_name="UTC", now=NOW)


async def packet(tmp_path, *, claims=True, checkpoints=1, extra=()):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}],
                         derive_claims=claims)
    if claims:
        assert await SourceClaimProjection(ledger).process_one(_Claims())
    history = [{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}, *extra]
    for n in range(checkpoints):
        # A later checkpoint carries the earlier messages again, and whatever came after them.
        checkpoint(ledger, history + [{"role": "user", "content": f"Later message {n}."}])
    collected = await collect_sources(ledger, query=PROBE, contact_id="owner", session_id=SESSION)
    return await select_memory(collected, query=PROBE, selector=RecallSelector(), timezone_name="UTC", now=NOW)


def test_a_checkpoint_copy_has_its_originals_hash(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, derive_claims=False,
                         messages=[{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}])
    checkpoint(ledger, [{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}])
    hits = ledger.search_sources("quarterly review studio", contact_id="owner", session_id=SESSION, limit=10)
    hashes = {(hit["scope"], hit["role"]): hit["source_message_hash"] for hit in hits}
    assert set(hashes) == {("person", "user"), ("person", "assistant"), ("session", "user"), ("session", "assistant")}
    assert hashes["session", "assistant"] == hashes["person", "assistant"] == source_message_hash(
        SESSION, {"role": "assistant", "content": REPLY})
    assert hashes["session", "user"] == hashes["person", "user"] != hashes["person", "assistant"]


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", [1, 3])
async def test_in_its_own_session_a_reply_is_still_never_shown_alone(tmp_path, monkeypatch, checkpoints):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    packet_ = await packet(tmp_path, checkpoints=checkpoints)
    assert "on Tuesday at 09:15 in the studio two" in packet_.content
    assert REPLY not in packet_.content            # the reply rule hid the original; its copies go with it
    # The person's words come as the complete message recalled with its claims; their copies' remainders go with
    # the message's own.
    assert not any(row.get("scope") == "session" for row in packet_.selected)


@pytest.mark.asyncio
async def test_in_its_own_session_a_paired_reply_comes_once_beside_its_input(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    packet_ = await packet(tmp_path, claims=False, checkpoints=2)
    assert [row["kind"] for row in packet_.selected].count("conversation_pair") == 1
    assert packet_.content.count("Tue Sep 29 09:15") == 1
    assert packet_.content.count("on Tuesday at 09:15 in the studio two") == 1
    assert not any(row.get("scope") == "session" for row in packet_.selected)


@pytest.mark.asyncio
async def test_what_only_a_checkpoint_holds_stays_recallable_once(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    packet_ = await packet(tmp_path, checkpoints=3, extra=[{"role": "assistant", "content": ONLY_IN_CHECKPOINT}])
    assert packet_.content.count(ONLY_IN_CHECKPOINT) == 1
    kept, = [row for row in packet_.selected if row.get("content") == ONLY_IN_CHECKPOINT]
    assert kept["scope"] == "session" and kept["role"] == "assistant"
    assert REPLY not in packet_.content


def test_only_a_copy_with_the_same_words_as_a_person_scope_candidate_is_dropped():
    from colonymind.memory.recall import collapse_checkpoint_copies, drop_checkpoint_copies

    def hit(turn, scope, role, content, message=None, session=SESSION):
        return {"turn_id": turn, "role": role, "content": content, "contact_id": "owner", "session_id": session,
                "scope": scope, "source_message_hash": source_message_hash(
                    session, {"role": role, "content": message or content})}

    long_reply = "First part of a long reply. " * 3 + "Second part of a long reply."
    rows = source_candidates([
        hit("t-1", "person", "assistant", REPLY),
        hit("checkpoint:a", "session", "assistant", REPLY),                   # the original carries it: dropped
        hit("checkpoint:b", "session", "assistant", REPLY),
        hit("t-2", "person", "assistant", "First part of a long reply.", long_reply),
        hit("checkpoint:a", "session", "assistant", "Second part of a long reply.", long_reply),  # other words: kept
        hit("checkpoint:a", "session", "assistant", "part of a long reply.", long_reply),  # inside, not equal: kept
        hit("checkpoint:a", "session", "assistant", ONLY_IN_CHECKPOINT),      # no original: kept, once
        hit("checkpoint:b", "session", "assistant", ONLY_IN_CHECKPOINT),
        hit("checkpoint:c", "session", "assistant", REPLY, session="s-2"),     # another session's words: kept
    ])
    kept = collapse_checkpoint_copies(drop_checkpoint_copies(rows))
    assert [(row["source_turn_id"], row["content"]) for row in kept] == [
        ("t-1", REPLY), ("t-2", "First part of a long reply."),
        ("checkpoint:a", "Second part of a long reply."), ("checkpoint:a", "part of a long reply."),
        ("checkpoint:a", ONLY_IN_CHECKPOINT), ("checkpoint:c", REPLY)]
    # A copy of the complete message recalled with its claims goes; a fragment of it is ranked in its own window.
    complete = {"id": "message-source:x", "kind": "source_quote", "scope": "person", "content": long_reply,
                "source_turn_id": "t-2", "source_message_hash": rows[3]["source_message_hash"]}
    whole = source_candidates([hit("checkpoint:a", "session", "assistant", long_reply)])
    assert drop_checkpoint_copies(whole, originals=[complete]) == []
    assert [row["content"] for row in drop_checkpoint_copies(rows[4:5], originals=[complete])] == [
        "Second part of a long reply."]
    # A copy goes only when the candidate carrying its words also carries whatever can correct it.
    digest = rows[0]["source_message_hash"]
    corrected = {("checkpoint:a", digest): frozenset({("correction", "note-1")})}
    assert [row["source_turn_id"] for row in drop_checkpoint_copies(rows[:3], membership=corrected)] == [
        "t-1", "checkpoint:a"]
    named = frozenset({("source", "report", "v1")})
    assert [row["source_turn_id"] for row in drop_checkpoint_copies(rows[:3], membership={
        ("checkpoint:a", digest): named})] == ["t-1", "checkpoint:a"]
    assert [row["source_turn_id"] for row in drop_checkpoint_copies(rows[:3], membership={
        ("t-1", digest): named, ("checkpoint:a", digest): named})] == ["t-1"]
    # Of identical copies, a later one stays only when it can carry a correction the ones before it cannot.
    copies = rows[6:8]
    assert [row["source_turn_id"] for row in collapse_checkpoint_copies(copies)] == ["checkpoint:a"]
    only = copies[0]["source_message_hash"]
    assert [row["source_turn_id"] for row in collapse_checkpoint_copies(copies, membership={
        ("checkpoint:b", only): named})] == ["checkpoint:a", "checkpoint:b"]
    assert [row["source_turn_id"] for row in collapse_checkpoint_copies(copies, membership={
        ("checkpoint:a", only): named, ("checkpoint:b", only): named})] == ["checkpoint:a"]


CORRECTION = "Moved to Wednesday at 10:00."


@pytest.mark.asyncio
async def test_a_correction_of_a_checkpoints_copy_stays_with_its_words(tmp_path, monkeypatch):
    # The owner corrected the words as a checkpoint recorded them; the person-scope original has no correction.
    # Dropping the copy for the original would lose the correction, and the original would stand unqualified.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}],
                         derive_claims=False)
    copy = checkpoint(ledger, [{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY},
                               {"role": "user", "content": "Later message."}])
    correct(ledger, copy, "Tuesday at 09:15", CORRECTION)
    packet_ = await recall(ledger)
    assert CORRECTION in packet_.content
    corrected, = [row for row in packet_.selected if row.get("epistemic_state") == "correction_evidence"]
    assert copy in corrected["source_turn_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected", [0, 1])
async def test_identical_checkpoint_copies_keep_their_own_corrections(tmp_path, monkeypatch, corrected):
    # Two checkpoints hold the same words and no person-scope source does; only one copy was corrected.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    copies = [checkpoint(ledger, [{"role": "user", "content": SAID}, {"role": "user", "content": f"Later {n}."}])
              for n in range(2)]
    order = [hit["turn_id"] for hit in ledger.search_sources(
        "quarterly review studio", contact_id="owner", session_id=SESSION, limit=10) if hit["content"] == SAID]
    assert sorted(order) == sorted(copies)
    correct(ledger, order[corrected], "Tuesday at 09:15", CORRECTION)
    packet_ = await recall(ledger)
    assert CORRECTION in packet_.content


@pytest.mark.asyncio
async def test_a_copy_that_names_a_corrected_source_keeps_that_correction(tmp_path, monkeypatch):
    # The checkpoint's copy of the reply names a source it was given; the original reply names none. A correction
    # of that source reaches the reply through the copy only.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("report", contact_id="owner", session_id="s-0", derive_claims=False,
                         messages=[{"role": "user", "content": "The booking desk confirmed the slot for Tuesday."}])
    correct(ledger, "report", "for Tuesday", CORRECTION)
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": SAID}, {"role": "assistant", "content": REPLY}],
                         derive_claims=False)
    report, = ledger.source_references(["report"], contact_id="owner", session_id=SESSION)
    checkpoint(ledger, [{"role": "user", "content": SAID},
                        {"role": "assistant", "content": REPLY, "_supplied_sources": [report]},
                        {"role": "user", "content": "Later message."}])
    packet_ = await recall(ledger)
    assert CORRECTION in packet_.content


@pytest.mark.asyncio
async def test_a_copy_in_a_ranking_window_of_its_own_stays_recallable(tmp_path, monkeypatch):
    # The reranker reads a candidate's first RERANK_DOCUMENT_CHARS characters. The answer sits past 1,800 characters
    # of the person's message: inside the person's first chunk (2,000 characters), outside what the reranker reads
    # of it. The checkpoint's copy returned the overlapping tail as a chunk of its own, the only window with the
    # answer in it; its words are inside the person's chunk, but not the same words, so it stays.
    answer = "The storage locker code is 4417."
    words = ("Packing notes for the move. " * 70)[:1805] + " " + answer + " Done."
    assert RERANK_DOCUMENT_CHARS < 1800 < words.index(answer) and len(words) <= 2000

    class Reranker:
        def calibration_metadata(self):
            return {"model": "neutral-fixture", "format_version": "v1"}

        async def rerank(self, query, documents, top_k):
            return [{"index": i, "score": .95 if "4417" in text else .10} for i, text in enumerate(documents)]

    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, derive_claims=False,
                         occurred_at="2026-09-25T12:00:04+00:00", messages=[{"role": "user", "content": words}])
    checkpoint(ledger, [{"role": "user", "content": words}, {"role": "user", "content": "Later message."}])
    hits = ledger.search_sources("storage locker code packing", contact_id="owner", session_id=SESSION, limit=10)
    person, = [hit for hit in hits if hit["scope"] == "person" and hit["content"] == words]
    tail, = [hit for hit in hits if hit["scope"] == "session" and hit["content"] == words[1800:]]
    assert person["source_message_hash"] == tail["source_message_hash"]
    reranker = Reranker()
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", ".67")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", calibration_fingerprint(
        provider_calibration_metadata(reranker)))
    selector = RecallSelector(reranker.rerank, calibration_metadata=lambda: provider_calibration_metadata(reranker))
    packet_ = await select_memory(CollectedSources(ledger, "owner", SESSION, None, hits=[person, tail],
                                                   lexical_hits=[person, tail]),
                                  query="What is the storage locker code?", selector=selector,
                                  timezone_name="UTC", now=NOW)
    assert answer in packet_.content
    assert [row["rerank_score"] for row in packet_.selected] == [.95]


@pytest.mark.asyncio
@pytest.mark.parametrize("days", [("25", "26"), ("26", "25")])
async def test_a_dated_question_finds_the_checkpoint_copy_from_its_day(tmp_path, monkeypatch, days):
    # Only checkpoints hold these words, recorded on two days. Each copy's own date decides whether it answers a
    # question about a day before identical copies occupy one slot; either retrieval order finds the earlier copy.
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    for n, day in enumerate(days):
        checkpoint(ledger, [{"role": "assistant", "content": ONLY_IN_CHECKPOINT},
                            {"role": "user", "content": f"Later message {n}."}],
                   occurred_at=f"2026-09-{day}T13:00:00+00:00")
    query = "What happened on 2026-09-25 with the proof of the quarterly review poster?"
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION)
    packet_ = await select_memory(collected, query=query, selector=RecallSelector(), timezone_name="UTC",
                                  now=datetime(2026, 9, 27, 12, tzinfo=timezone.utc))
    kept, = [row for row in packet_.selected if row.get("content") == ONLY_IN_CHECKPOINT]
    assert kept["occurred_at"] == "2026-09-25T13:00:00+00:00"
    assert kept["validity_status"] == "source_occurrence_only"
    # Without a date both copies answer, and one slot holds them.
    query = "What goes to the printer with the quarterly review poster?"
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION)
    packet_ = await select_memory(collected, query=query, selector=RecallSelector(), timezone_name="UTC", now=NOW)
    assert packet_.content.count(ONLY_IN_CHECKPOINT) == 1


class _AnswerReranker:
    """Scores a document that holds the answer high and every other one low, and keeps what it was sent."""

    def __init__(self, answer):
        self.answer, self.documents = answer, []

    def calibration_metadata(self):
        return {"model": "neutral-fixture", "format_version": "v1"}

    async def rerank(self, query, documents, top_k):
        self.documents.append(list(documents))
        return [{"index": i, "score": .95 if self.answer in text else .10} for i, text in enumerate(documents)]


def _calibrated(monkeypatch, reranker):
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", ".5")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION", calibration_fingerprint(
        provider_calibration_metadata(reranker)))
    return RecallSelector(reranker.rerank, calibration_metadata=lambda: provider_calibration_metadata(reranker))


LONG_INPUT = (("Moving checklist, part one of the packing notes, nothing decided yet. " * 25)[:1700]
              + " Where should the spare keys go?")
KEYS = "The spare keys go in the blue tin on the hall shelf."


@pytest.mark.asyncio
async def test_a_copy_of_a_reply_its_pair_ranks_past_the_window_stays_a_candidate(tmp_path, monkeypatch):
    # The person's message runs past the reranker's window, so the pair's text (input first) shows the reranker
    # none of the reply. The checkpoint's copy of the reply is the one window with the answer in it: had it gone
    # with its original, the calibrated cutoff would leave nothing. It stays a candidate of its own.
    assert len("User input:\n" + LONG_INPUT) > RERANK_DOCUMENT_CHARS
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, derive_claims=False,
                         occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS}])
    checkpoint(ledger, [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS},
                        {"role": "user", "content": "Later message."}])
    reranker = _AnswerReranker("blue tin")
    query = "Where did we say the spare keys go?"
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION)
    packet_ = await select_memory(collected, query=query, selector=_calibrated(monkeypatch, reranker),
                                  timezone_name="UTC", now=NOW)
    assert KEYS in packet_.content
    kept, = packet_.selected
    assert (kept["scope"], kept["role"], kept["rerank_score"]) == ("session", "assistant", .95)
    # The reranker is sent what the calibrated format sends: one window per candidate, the pair's text cut at the
    # window (none of the reply in it), and the copy's own words.
    documents, = reranker.documents
    pair = "User input:\n" + LONG_INPUT + "\nAssistant response:\n" + KEYS
    assert pair[:RERANK_DOCUMENT_CHARS] in documents and KEYS in documents
    assert all(len(document) <= RERANK_DOCUMENT_CHARS for document in documents)


def test_a_copy_goes_only_when_the_pair_its_original_joins_shows_the_copys_window():
    from colonymind.memory.recall import drop_checkpoint_copies, pair_conversation_candidates

    def kept(said, reply, *, input_recalled=True):
        messages = [{"role": "user", "content": said}, {"role": "assistant", "content": reply}]
        digests = [source_message_hash(SESSION, message) for message in messages]
        hits = [{"turn_id": turn, "role": message["role"], "content": message["content"], "contact_id": "owner",
                 "session_id": SESSION, "scope": scope, "source_message_hash": digest}
                for turn, scope in (("t-1", "person"), ("checkpoint:a", "session"))
                for message, digest in zip(messages, digests)]
        rows = source_candidates(hits if input_recalled else hits[1:])
        paired = pair_conversation_candidates(rows, {("t-1", digests[1]): ("t-1", digests[0])},
                                              {"t-1": {"messages_json": json.dumps(messages)}})
        return [(row["kind"], row.get("scope"), row.get("role"))
                for row in drop_checkpoint_copies(paired, originals=rows)]

    pair = ("conversation_pair", "person", None)
    # The pair's window shows both messages whole: both copies go with their originals.
    assert kept(SAID, REPLY) == [pair]
    # After a long input the pair's window shows none of the reply, and only the input's first 1,488 characters
    # ("User input:" and a line break come first): each copy stays a window of its own.
    assert kept(LONG_INPUT, KEYS) == [pair, ("source_quote", "session", "user"), ("source_quote", "session", "assistant")]
    # A long reply is shown only in part.
    assert kept(SAID, "Unpacking notes, box by box. " * 60) == [pair, ("source_quote", "session", "assistant")]
    # A reply the reply rule hides (its input was not recalled beside it) takes its copy with it, as before.
    assert kept(SAID, REPLY, input_recalled=False) == [("source_quote", "session", "user")]


MOVED = "The spare keys now go in the red drawer."


def _long_conversation_with_a_checkpoint(ledger):
    ledger.record_source("owner-1:t-1", contact_id="owner", session_id=SESSION, derive_claims=False,
                         occurred_at="2026-09-25T12:00:04+00:00",
                         messages=[{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS}])
    checkpoint(ledger, [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS},
                        {"role": "user", "content": "Later message."}])


@pytest.mark.asyncio
async def test_a_kept_copy_carries_the_corrections_of_its_original(tmp_path, monkeypatch):
    # The owner corrected the reply where recall showed it: in the original. The pair its original joins shows the
    # reranker none of the reply, so the checkpoint's copy stays a window of its own; it must not come back as the
    # obsolete answer without the correction.
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    _long_conversation_with_a_checkpoint(ledger)
    correct(ledger, "owner-1:t-1", "blue tin on the hall shelf", MOVED)
    reranker = _AnswerReranker("blue tin")
    query = "Where did we say the spare keys go?"
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION)
    packet_ = await select_memory(collected, query=query, selector=_calibrated(monkeypatch, reranker),
                                  timezone_name="UTC", now=NOW)
    kept, = packet_.selected
    assert (kept["scope"], kept["role"], kept["epistemic_state"]) == ("session", "assistant", "correction_evidence")
    assert KEYS in packet_.content and MOVED in packet_.content
    assert "owner-1:t-1" in {ref["source_id"] for ref in kept["_annotation_source_refs"]}
    # It is ranked as any corrected candidate is: the correction first, then the words.
    documents, = reranker.documents
    assert any(document.startswith("Attributed correction (not independently verified):\n" + MOVED)
               and document.endswith(KEYS) for document in documents)


@pytest.mark.asyncio
@pytest.mark.parametrize("pair_first", [True, False])
async def test_a_kept_copy_goes_when_the_pair_it_was_kept_for_is_in_the_packet(tmp_path, monkeypatch, pair_first):
    # The copies stay because the pair shows the reranker only part of them. When the pair clears the cutoff too, it
    # shows the model both messages whole, so neither copy adds anything: each message comes once, the reply beside
    # the message it answers, whichever of them ranked first.
    class Reranker(_AnswerReranker):
        async def rerank(self, query, documents, top_k):
            self.documents.append(list(documents))
            high, low = (.95, .9) if pair_first else (.9, .95)
            return [{"index": i, "score": high if text.startswith("User input:") else low}
                    for i, text in enumerate(documents)]

    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    _long_conversation_with_a_checkpoint(ledger)
    reranker = Reranker("")
    query = "Where did we say the spare keys go?"
    collected = await collect_sources(ledger, query=query, contact_id="owner", session_id=SESSION)
    packet_ = await select_memory(collected, query=query, selector=_calibrated(monkeypatch, reranker),
                                  timezone_name="UTC", now=NOW)
    # All three were sent and all three cleared the cutoff.
    documents, = reranker.documents
    assert KEYS in documents and len(documents) == 3
    assert [row["kind"] for row in packet_.selected] == ["conversation_pair"]
    assert packet_.content.count(KEYS) == 1
    assert packet_.content.count(LONG_INPUT[-200:]) == 1


def test_a_pair_takes_the_place_of_the_copies_kept_for_its_window_only_when_it_fits_whole():
    from colonymind.memory.recall import drop_checkpoint_copies, pack_memory_context, pair_conversation_candidates

    messages = [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": KEYS}]
    digests = [source_message_hash(SESSION, message) for message in messages]
    rows = source_candidates([
        {"turn_id": turn, "role": message["role"], "content": message["content"], "contact_id": "owner",
         "session_id": SESSION, "scope": scope, "source_message_hash": digest}
        for turn, scope in (("t-1", "person"), ("checkpoint:a", "session"))
        for message, digest in zip(messages, digests)])
    pair, said, reply = drop_checkpoint_copies(pair_conversation_candidates(
        rows, {("t-1", digests[1]): ("t-1", digests[0])}, {"t-1": {"messages_json": json.dumps(messages)}}),
        originals=rows)
    # Each copy names its original's source, for its corrections, and the message the pair cut off.
    assert [copy["source_turn_ids"] for copy in (said, reply)] == [["checkpoint:a", "t-1"]] * 2
    assert [copy["_ranking_window_of"] for copy in (said, reply)] == [[["t-1", digest]] for digest in digests]
    # Whichever ranked first, the pair shows both messages whole and the copies add nothing beside it.
    for order in ([pair, reply, said], [reply, said, pair], [reply, pair, said]):
        selected, content = pack_memory_context(order, max_chars=6000)
        assert [row["kind"] for row in selected] == ["conversation_pair"]
        assert content.count(KEYS) == 1
    # A pair that does not fit whole leaves in place the copy that ranked before it.
    selected, content = pack_memory_context([reply, pair], max_chars=1000)
    assert selected[0]["id"] == reply["id"] and content.count(KEYS) == 1
