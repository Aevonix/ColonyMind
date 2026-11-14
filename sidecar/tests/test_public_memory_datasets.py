"""The public memory datasets (``benchmarks/public_memory/datasets``): pinned downloads, the adapters that turn
upstream files into sessions, questions and gold turns, and the seeded subsets.

Every test here runs on small files written in the test. The pinned upstream files are never committed; the tests
at the end read them only when ``COLONYMIND_BENCH_DATA_DIR`` holds them (``python -m
benchmarks.public_memory.datasets fetch`` puts them there), and skip otherwise."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

DATASETS = Path(__file__).resolve().parents[2] / "benchmarks" / "public_memory" / "datasets"
PACKAGE = "public_memory_datasets_under_test"


def _module(name: str):
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, DATASETS / "__init__.py",
                                                      submodule_search_locations=[str(DATASETS)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{PACKAGE}.{name}")


def _pin(name: str, data: bytes, url: str = "https://example.invalid/data"):
    pinned = _module("pinned")
    return pinned.Pin(name=name, url=url, sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def _opener(data: bytes, calls: list):
    def opener(url, timeout=None):
        calls.append(url)
        return io.BytesIO(data)
    return opener


# -- pinned downloads ---------------------------------------------------------------------------------------------


def test_a_cached_file_whose_hash_differs_from_the_pin_is_refused_and_left_alone(tmp_path):
    pinned = _module("pinned")
    (tmp_path / "data.json").write_bytes(b"changed upstream")
    with pytest.raises(pinned.PinError, match="SHA-256"):
        pinned.fetch(_pin("data.json", b"what was pinned"), tmp_path, download=False)
    assert (tmp_path / "data.json").read_bytes() == b"changed upstream"


def test_a_download_whose_hash_differs_from_the_pin_is_refused_and_nothing_is_kept(tmp_path):
    pinned = _module("pinned")
    calls = []
    with pytest.raises(pinned.PinError, match="SHA-256"):
        pinned.fetch(_pin("data.json", b"what was pinned"), tmp_path, opener=_opener(b"tampered", calls))
    assert calls == ["https://example.invalid/data"]
    assert sorted(tmp_path.iterdir()) == []


def test_a_matching_download_is_kept_and_read_from_the_cache_afterwards(tmp_path):
    pinned = _module("pinned")
    pin, calls = _pin("data.json", b"pinned bytes"), []
    path = pinned.fetch(pin, tmp_path, opener=_opener(b"pinned bytes", calls))
    assert path == tmp_path / "data.json" and path.read_bytes() == b"pinned bytes"
    assert pinned.fetch(pin, tmp_path, opener=_opener(b"never read", calls)) == path
    assert len(calls) == 1
    assert sorted(item.name for item in tmp_path.iterdir()) == ["data.json"]


def test_a_missing_file_is_refused_when_downloads_are_off(tmp_path):
    pinned = _module("pinned")
    with pytest.raises(pinned.PinError, match="not in"):
        pinned.fetch(_pin("data.json", b"pinned bytes"), tmp_path, download=False)


def test_the_cache_is_the_named_directory_then_the_environment_then_the_user_cache(tmp_path, monkeypatch):
    pinned = _module("pinned")
    monkeypatch.setenv("COLONYMIND_BENCH_DATA_DIR", str(tmp_path / "from-env"))
    assert pinned.cache_dir(tmp_path / "named") == tmp_path / "named"
    assert pinned.cache_dir() == tmp_path / "from-env"
    monkeypatch.delenv("COLONYMIND_BENCH_DATA_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert pinned.cache_dir() == tmp_path / "cache" / "colonymind" / "public-datasets"


# -- LongMemEval-S -----------------------------------------------------------------------------------------------

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def stamp(text: str) -> str:
    """``2023-05-20 08:00`` in LongMemEval's own form, ``2023/05/20 (Sat) 08:00``."""
    moment = datetime.strptime(text, "%Y-%m-%d %H:%M")
    return f"{moment:%Y/%m/%d} ({WEEKDAYS[moment.weekday()]}) {moment:%H:%M}"


def turn(role: str, content: str, has_answer=None) -> dict:
    return {"role": role, "content": content} if has_answer is None else \
        {"role": role, "content": content, "has_answer": has_answer}


def chat(*marks) -> list:
    """A session of user/assistant pairs; a true mark flags that turn as evidence."""
    turns = []
    for number, mark in enumerate(marks):
        turns.append(turn("user" if number % 2 == 0 else "assistant", f"turn {number}", mark))
    return turns


def lme_item(qid: str, qtype: str, asked: str, sessions: list, answers: list, answer="an answer") -> dict:
    return {"question_id": qid, "question_type": qtype, "question": f"question {qid}", "answer": answer,
            "question_date": stamp(asked), "haystack_session_ids": [sid for sid, _, _ in sessions],
            "haystack_dates": [stamp(when) for _, when, _ in sessions],
            "haystack_sessions": [turns for _, _, turns in sessions], "answer_session_ids": answers}


def lme_file(tmp_path: Path, items: list) -> Path:
    path = tmp_path / "longmemeval_s_cleaned.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def question(dataset, qid: str):
    return next(item for item in dataset.questions if item.id == qid)


def sessions_of(dataset, qid: str):
    return dataset.conversations[question(dataset, qid).conversation].sessions


def test_longmemeval_expected_counts_are_the_published_ones_and_a_different_count_is_refused(tmp_path):
    lme, records = _module("longmemeval_s"), _module("records")
    assert lme.EXPECTED_TYPES == {"single-session-user": 70, "single-session-assistant": 56,
                                  "single-session-preference": 30, "multi-session": 133,
                                  "temporal-reasoning": 133, "knowledge-update": 78}
    assert sum(lme.EXPECTED_TYPES.values()) == 500 and lme.EXPECTED_ABSTENTION == 30
    dataset = lme.parse(lme_file(tmp_path, [
        lme_item("q1", "multi-session", "2023-05-22 10:00", [("s1", "2023-05-20 08:00", chat(True))], ["s1"]),
        lme_item("q2_abs", "multi-session", "2023-05-22 10:00", [("s2", "2023-05-20 08:00", chat())], ["s2"]),
    ]))
    assert dataset.counts() == {"multi-session": 2} and dataset.abstention_ids() == ("q2_abs",)
    records.verify_counts(dataset, {"multi-session": 2}, abstention=1)
    with pytest.raises(records.DatasetError, match="multi-session"):
        records.verify_counts(dataset, {"multi-session": 3}, abstention=1)
    with pytest.raises(records.DatasetError, match="abstention"):
        records.verify_counts(dataset, {"multi-session": 2}, abstention=2)


def test_longmemeval_sessions_replay_in_time_order_whatever_order_the_file_lists_them(tmp_path):
    lme = _module("longmemeval_s")
    dataset = lme.parse(lme_file(tmp_path, [
        lme_item("mixed", "multi-session", "2023-05-23 10:00", [
            ("c", "2023-05-21 09:00", chat()), ("a", "2023-05-20 08:00", chat()),
            ("b", "2023-05-20 08:00", chat()), ("d", "2023-05-22 10:00", chat(True))], ["d"]),
        lme_item("ordered", "multi-session", "2023-05-23 10:00", [
            ("a", "2023-05-20 08:00", chat()), ("b", "2023-05-21 08:00", chat(True))], ["b"]),
    ]))
    mixed = sessions_of(dataset, "mixed")
    assert [item.key for item in mixed] == ["a", "b", "c", "d"]   # a tie keeps the file's order
    assert [item.position for item in mixed] == [1, 2, 0, 3]
    assert all(earlier.at <= later.at for earlier, later in zip(mixed, mixed[1:]))
    assert mixed[0].at == datetime(2023, 5, 20, 8, 0, tzinfo=timezone.utc)
    assert dataset.findings["reordered"] == ("mixed",) and dataset.findings["tied_sessions"] == ("mixed",)


def test_no_longmemeval_session_falls_after_the_time_its_question_is_asked(tmp_path):
    lme = _module("longmemeval_s")
    dataset = lme.parse(lme_file(tmp_path, [
        lme_item("late", "temporal-reasoning", "2023-05-22 10:00", [
            ("a", "2023-05-22 17:30", chat(True)), ("b", "2023-05-21 09:00", chat())], ["a"]),
        lme_item("midnight", "knowledge-update", "2023-05-22 10:00", [
            ("a", "2023-05-22 23:59", chat(True))], ["a"]),
        lme_item("plain", "knowledge-update", "2023-05-22 10:00", [("a", "2023-05-22 09:00", chat(True))], ["a"]),
    ]))
    late, midnight, plain = (question(dataset, qid) for qid in ("late", "midnight", "plain"))
    assert late.asked_at == datetime(2023, 5, 22, 17, 31, tzinfo=timezone.utc)
    assert late.date_text == "2023/05/22 (Mon) 10:00"   # what the reader is told stays the dataset's
    assert midnight.asked_at == datetime(2023, 5, 22, 23, 59, 59, tzinfo=timezone.utc)
    assert plain.asked_at == datetime(2023, 5, 22, 10, 0, tzinfo=timezone.utc)
    for item in dataset.questions:
        assert all(session.at <= item.asked_at for session in sessions_of(dataset, item.id))
        assert item.asked_at.date() == datetime.strptime(item.date_text[:10], "%Y/%m/%d").date()
    assert dataset.findings["asked_after_last_session"] == ("late", "midnight")


def test_a_longmemeval_session_on_a_later_day_than_its_question_is_refused(tmp_path):
    lme, records = _module("longmemeval_s"), _module("records")
    path = lme_file(tmp_path, [lme_item("q", "temporal-reasoning", "2023-05-22 10:00", [
        ("a", "2023-05-23 08:00", chat(True))], ["a"])])
    with pytest.raises(records.DatasetError, match="after the question"):
        lme.parse(path)


def test_a_longmemeval_stamp_whose_weekday_disagrees_with_its_date_is_refused(tmp_path):
    lme, records = _module("longmemeval_s"), _module("records")
    item = lme_item("q", "multi-session", "2023-05-22 10:00", [("a", "2023-05-20 08:00", chat(True))], ["a"])
    item["haystack_dates"] = ["2023/05/20 (Sun) 08:00"]   # the 20th was a Saturday
    with pytest.raises(records.DatasetError, match="weekday"):
        lme.parse(lme_file(tmp_path, [item]))


def test_longmemeval_gold_turns_are_the_marked_turns_and_abstention_is_the_id_suffix(tmp_path):
    lme = _module("longmemeval_s")
    records = _module("records")
    dataset = lme.parse(lme_file(tmp_path, [
        lme_item("q", "multi-session", "2023-05-22 10:00", [
            ("noise", "2023-05-19 08:00", chat()),
            ("first", "2023-05-20 08:00", chat(None, False, True)),
            ("second", "2023-05-21 08:00", chat(True, True))], ["second", "first"], answer=3),
        lme_item("q_abs", "knowledge-update", "2023-05-22 10:00", [
            ("related", "2023-05-20 08:00", chat()), ("noise", "2023-05-21 08:00", chat())], ["related"]),
    ]))
    answered, unanswerable = question(dataset, "q"), question(dataset, "q_abs")
    assert answered.gold_sessions == ("first", "second")
    assert answered.evidence == (records.Evidence("first", 2), records.Evidence("second", 0),
                                 records.Evidence("second", 1))
    assert answered.answer == "3" and not answered.abstention
    assert unanswerable.abstention and unanswerable.gold_sessions == ("related",) and unanswerable.evidence == ()
    assert dataset.findings["unmarked_answer_sessions"] == ("q_abs",)
    first = next(item for item in sessions_of(dataset, "q") if item.key == "first")
    assert [(item.index, item.role) for item in first.turns] == [(0, "user"), (1, "assistant"), (2, "user")]


def test_a_repeated_longmemeval_session_keeps_each_occurrence_under_its_own_key(tmp_path):
    lme = _module("longmemeval_s")
    dataset = lme.parse(lme_file(tmp_path, [lme_item("q", "single-session-user", "2023-05-22 10:00", [
        ("twice", "2023-05-20 08:00", chat()), ("gold", "2023-05-20 09:00", chat(True)),
        ("twice", "2023-05-21 08:00", chat())], ["gold"])]))
    sessions = sessions_of(dataset, "q")
    assert [(item.key, item.source_id) for item in sessions] == [
        ("twice", "twice"), ("gold", "gold"), ("twice#2", "twice")]
    assert dataset.findings["repeated_sessions"] == ("q",)



# -- BEAM ---------------------------------------------------------------------------------------------------------

BEAM_CATEGORIES = ("abstention", "contradiction_resolution", "event_ordering", "information_extraction",
                   "instruction_following", "knowledge_update", "multi_session_reasoning", "preference_following",
                   "summarization", "temporal_reasoning")


def beam_row(conversation_id: str, sessions: list, probing: dict) -> dict:
    """One upstream row: sessions of (time anchor, turns); chat ids run on across the conversation."""
    chat_sessions, next_id = [], 0
    for anchor, turns in sessions:
        session = []
        for number, (role, content) in enumerate(turns):
            session.append({"content": content, "id": next_id, "index": None, "question_type": None, "role": role,
                            "time_anchor": anchor if number == 0 else None})
            next_id += 1
        chat_sessions.append(session)
    return {"conversation_id": conversation_id, "chat": chat_sessions, "probing_questions": repr(probing)}


def beam_file(tmp_path: Path, rows: list) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "beam-100K.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def pairs(count: int) -> list:
    return [("user" if number % 2 == 0 else "assistant", f"turn {number}") for number in range(count)]


def test_beam_expected_counts_are_the_published_ones(tmp_path):
    beam, records = _module("beam"), _module("records")
    assert beam.EXPECTED_TYPES == {name: 40 for name in BEAM_CATEGORIES} and beam.EXPECTED_CONVERSATIONS == 20
    dataset = beam.parse(beam_file(tmp_path, [beam_row("1", [("March-15-2024", pairs(4))], {
        "abstention": [{"question": "q", "ideal_response": "nothing", "rubric": ["r"]}],
        "information_extraction": [{"question": "q", "answer": "a", "source_chat_ids": [2], "rubric": ["r"]}]})]))
    records.verify_counts(dataset, {"abstention": 1, "information_extraction": 1}, abstention=1)
    with pytest.raises(records.DatasetError, match="abstention"):
        records.verify_counts(dataset, {"abstention": 40, "information_extraction": 1})


def test_beam_sessions_take_their_day_from_the_time_anchor_and_replay_in_time_order(tmp_path):
    beam = _module("beam")
    dataset = beam.parse(beam_file(tmp_path, [
        beam_row("7", [("April-05-2024", pairs(2)), ("March-15-2024", pairs(2))],
                 {"information_extraction": [{"question": "q", "answer": "a", "source_chat_ids": [1]}]})]))
    sessions = dataset.conversations["7"].sessions
    assert [(item.key, item.position) for item in sessions] == [("s2", 1), ("s1", 0)]
    assert sessions[0].at == datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    assert [turn.ref for turn in sessions[0].turns] == ["2", "3"]
    assert dataset.findings["reordered"] == ("7",)


def test_a_beam_session_with_no_time_anchor_is_refused(tmp_path):
    beam, records = _module("beam"), _module("records")
    with pytest.raises(records.DatasetError, match="time anchor"):
        beam.parse(beam_file(tmp_path, [beam_row("1", [(None, pairs(2))], {})]))


def test_beam_questions_are_asked_the_day_after_the_last_session(tmp_path):
    beam = _module("beam")
    sessions = [("March-15-2024", pairs(2)), ("April-25-2024", pairs(2))]
    dataset = beam.parse(beam_file(tmp_path, [beam_row("1", sessions, {
        "temporal_reasoning": [{"question": "q", "answer": "a", "source_chat_ids": {"first_event": [0],
                                                                                     "second_event": [2]}}]})]))
    item = dataset.questions[0]
    assert item.asked_at == datetime(2024, 4, 26, 12, 0, tzinfo=timezone.utc)
    assert item.date_text is None and item.conversation == "1" and item.id == "1:temporal_reasoning:0"


def test_beam_evidence_ids_become_gold_turns_in_every_upstream_form(tmp_path):
    beam, records = _module("beam"), _module("records")
    sessions = [("March-15-2024", pairs(4)), ("April-05-2024", pairs(4))]
    dataset = beam.parse(beam_file(tmp_path, [beam_row("1", sessions, {
        "abstention": [{"question": "q", "ideal_response": "not in the chat", "rubric": ["r"]}],
        "event_ordering": [{"question": "q", "answer": "a", "source_chat_ids": [[4, 6], [0]]}],
        "knowledge_update": [{"question": "q", "answer": "new", "source_chat_ids": {"original_info": [2],
                                                                                    "updated_info": [6]}}],
        "summarization": [{"question": "q", "ideal_summary": "s", "source_chat_ids": []}],
        "instruction_following": [{"question": "q", "expected_compliance": "c", "source_chat_ids": [0]}],
        "contradiction_resolution": [{"question": "q", "ideal_answer": "which?", "rubric": ["ask"],
                                      "source_chat_ids": {"first_statement": [0], "second_statement": [4]}}],
    })]))
    by_type = {item.type: item for item in dataset.questions}
    assert by_type["event_ordering"].evidence == (records.Evidence("s1", 0), records.Evidence("s2", 0),
                                                  records.Evidence("s2", 2))
    assert by_type["event_ordering"].gold_sessions == ("s1", "s2")
    assert by_type["knowledge_update"].evidence == (records.Evidence("s1", 2, "original_info"),
                                                    records.Evidence("s2", 2, "updated_info"))
    assert by_type["contradiction_resolution"].evidence == (records.Evidence("s1", 0, "first_statement"),
                                                            records.Evidence("s2", 0, "second_statement"))
    unanswerable = by_type["abstention"]
    assert unanswerable.abstention and unanswerable.evidence == () and unanswerable.answer == "not in the chat"
    assert not any(item.abstention for item in dataset.questions if item.type != "abstention")
    assert by_type["summarization"].answer == "s" and by_type["instruction_following"].answer == "c"
    assert by_type["contradiction_resolution"].answer == "which?"
    assert by_type["contradiction_resolution"].detail["rubric"] == ["ask"]
    assert dataset.findings["no_turn_evidence"] == ("1:summarization:0",)


def test_a_beam_evidence_id_that_names_no_turn_is_refused(tmp_path):
    beam, records = _module("beam"), _module("records")
    with pytest.raises(records.DatasetError, match="99"):
        beam.parse(beam_file(tmp_path, [beam_row("1", [("March-15-2024", pairs(2))], {
            "information_extraction": [{"question": "q", "answer": "a", "source_chat_ids": [99]}]})]))



# -- PrecisionMemBench --------------------------------------------------------------------------------------------


def belief(bid: str, **fields) -> dict:
    record = {"_id": bid, "user_id": "test-user", "type": "preference", "canonical_name": bid.replace("-", "_"),
              "aliases": [], "content": f"content of {bid}", "why_it_matters": "", "scope": ["domain:code"],
              "epistemic_status": "active", "confidence": 0.9, "pinned": False, "reinforcement_count": 1,
              "last_reinforced_at": "2026-10-01T00:00:00Z", "user_edited": False, "resolved_at": None,
              "superseded_by": None, "created_at": "2026-09-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z",
              "provenance": {"session_id": "seed", "turn_id": "seed"}}
    record.update(fields)
    return record


def pmb_files(tmp_path: Path, beliefs: list, cases: list, sessions: list) -> dict:
    pmb = _module("precisionmembench")
    paths = {}
    for key, data in (("beliefs", beliefs), ("cases", cases), ("sessions", sessions)):
        path = tmp_path / pmb.PINS[key].name
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[key] = path
    return paths


def test_precisionmembench_beliefs_and_cases_parse_into_typed_expectations(tmp_path):
    pmb = _module("precisionmembench")
    corpus = pmb.parse(pmb_files(tmp_path, [
        belief("b-db", pinned=True, type="decision"), belief("b-old", epistemic_status="superseded",
                                                             superseded_by="b-db"),
        belief("b-other", user_id="other-user")], [
        {"caseId": "c1", "category": "Supersession chain exclusion", "description": "d", "scope": ["domain:code"],
         "query": "which database?", "expect": {"pinnedFacts": {"mustInclude": ["b-db"]},
                                                "relevantBeliefs": {"shouldOnlyInclude": [], "mustExclude": ["b-old"]},
                                                "personaPrelude": {"nonEmpty": True, "contains": ["direct"]}}},
        {"caseId": "c2", "category": "Cold start behavior", "description": "d", "scope": ["domain:code"],
         "userId": "brand-new-user", "budget": {"maxBeliefs": 0}, "query": "q",
         "expect": {"personaPrelude": {"isNull": True}, "relevantBeliefs": {"mustExclude": ["b-other"]}}},
    ], []))
    old = next(item for item in corpus.beliefs if item.id == "b-old")
    assert old.status == "superseded" and old.superseded_by == "b-db" and old.user == "test-user"
    assert old.created_at == datetime(2026, 9, 1, tzinfo=timezone.utc)
    first, second = corpus.cases
    assert first.user == "test-user" and first.scope == ("domain:code",) and first.budget is None
    assert first.expect["pinnedFacts"].must_include == ("b-db",)
    assert first.expect["relevantBeliefs"].only == () and first.expect["relevantBeliefs"].must_exclude == ("b-old",)
    assert first.expect["pinnedFacts"].only is None
    assert first.expect["personaPrelude"].non_empty is True and first.expect["personaPrelude"].contains == ("direct",)
    assert second.user == "brand-new-user" and second.budget == {"maxBeliefs": 0}
    assert second.expect["personaPrelude"].is_null is True
    assert corpus.counts() == {"Supersession chain exclusion": 1, "Cold start behavior": 1}
    assert corpus.findings == {"unread_conditions": ()}


def test_a_precisionmembench_case_that_names_an_unknown_belief_is_refused(tmp_path):
    pmb, records = _module("precisionmembench"), _module("records")
    paths = pmb_files(tmp_path, [belief("b-db")], [
        {"caseId": "c1", "category": "Alias resolution", "description": "d", "scope": [], "query": "q",
         "expect": {"relevantBeliefs": {"mustInclude": ["b-missing"]}}}], [])
    with pytest.raises(records.DatasetError, match="b-missing"):
        pmb.parse(paths)


def test_precisionmembench_session_turns_keep_their_messages_and_what_they_create(tmp_path):
    pmb = _module("precisionmembench")
    created = belief("b-new", type="decision")
    corpus = pmb.parse(pmb_files(tmp_path, [belief("b-redis")], [], [
        {"caseId": "s1", "description": "d", "notes": "", "turns": [
            {"turnIndex": 0, "label": "establishes_topic", "scope": ["domain:code"], "topics": ["cache"],
             "turnSignal": "substantive", "userMessage": "u0", "assistantMessage": "a0",
             "expect": {"relevantBeliefs": {"shouldOnlyInclude": ["b-redis"]}}, "createBeliefAtTurn": created},
            {"turnIndex": 1, "label": "drift", "scope": ["domain:code"], "topics": ["ci"],
             "turnSignal": "substantive", "userMessage": "u1", "assistantMessage": "a1",
             "expect": {"relevantBeliefs": {"shouldOnlyInclude": [], "_note": "a note"},
                        "noiseCheck": {"mustNotSurface": ["b-new"]}},
             "updateBeliefAtTurn": {"beliefId": "b-redis", "addAliases": ["eviction"]}}]},
        {"caseId": "s2", "description": "d", "notes": "", "turns": [
            {"turnIndex": 0, "label": "implicit_continuation", "scope": ["domain:code"], "topics": ["cache"],
             "turnSignal": "substantive", "userMessage": "u", "assistantMessage": "a",
             "expect": {"noiseCheck": {"mustNotSurface": ["b-new"]},
                        "relevantBeliefs": {"pinnedFacts": {"mustInclude": ["b-redis"]}}}}]},
    ]))
    first, second = corpus.sessions
    assert [(item.index, item.user_message, item.assistant_message) for item in first.turns] == [
        (0, "u0", "a0"), (1, "u1", "a1")]
    assert first.turns[0].creates.id == "b-new" and first.turns[1].updates == {"beliefId": "b-redis",
                                                                              "addAliases": ["eviction"]}
    assert first.turns[1].expect["noiseCheck"].must_not_surface == ("b-new",)
    assert first.turns[1].expect["relevantBeliefs"].only == ()
    assert corpus.scored_turns() == 3
    assert corpus.findings == {"unread_conditions": ("s2:0",)}


def test_precisionmembench_expected_counts_are_the_pinned_fixtures(tmp_path):
    pmb, records = _module("precisionmembench"), _module("records")
    assert sum(pmb.EXPECTED_CATEGORIES.values()) == 77 and pmb.EXPECTED_BELIEFS == 35
    assert pmb.EXPECTED_SESSIONS == 2 and pmb.EXPECTED_SCORED_TURNS == 12
    corpus = pmb.parse(pmb_files(tmp_path, [belief("b-db")], [
        {"caseId": "c1", "category": "Alias resolution", "description": "d", "scope": [], "query": "q",
         "expect": {"relevantBeliefs": {"mustInclude": ["b-db"]}}}], []))
    with pytest.raises(records.DatasetError, match="Alias resolution"):
        records.verify_counts(corpus, {"Alias resolution": 23})



# -- seeded, stratified subsets -----------------------------------------------------------------------------------

LME_SHAPE = {"multi-session": 133, "temporal-reasoning": 133, "knowledge-update": 78, "single-session-user": 70,
             "single-session-assistant": 56, "single-session-preference": 30}


def synthetic(shape=LME_SHAPE, abstention_every=0, sha="0" * 64, reverse=False):
    records = _module("records")
    asked = datetime(2023, 5, 30, tzinfo=timezone.utc)
    questions = [records.Question(id=f"{kind}-{number}", conversation=f"{kind}-{number}", type=kind, question="",
                                  answer="", asked_at=asked, date_text=None,
                                  abstention=bool(abstention_every) and number % abstention_every == 0,
                                  gold_sessions=(), evidence=())
                 for kind, count in shape.items() for number in range(count)]
    if reverse:
        questions.reverse()
    return records.Dataset(name="synthetic", sha256=sha, conversations={}, questions=tuple(questions), findings={})


def test_a_subset_is_the_same_for_the_same_seed_whatever_order_the_questions_come_in():
    subsets = _module("subsets")
    first = subsets.draw(synthetic(), seed=7, size=60)
    again = subsets.draw(synthetic(reverse=True), seed=7, size=60)
    assert set(first.ids) == set(again.ids) and len(first.ids) == 60
    assert first.record() == subsets.draw(synthetic(), seed=7, size=60).record()


def test_a_different_seed_draws_a_different_subset():
    subsets = _module("subsets")
    assert set(subsets.draw(synthetic(), seed=1, size=60).ids) != set(subsets.draw(synthetic(), seed=2, size=60).ids)


def test_each_stratum_gets_its_share_of_the_size_by_largest_remainder():
    subsets = _module("subsets")
    assert subsets.draw(synthetic(), seed=1, size=60).strata == {
        "multi-session": 16, "temporal-reasoning": 16, "knowledge-update": 9, "single-session-user": 8,
        "single-session-assistant": 7, "single-session-preference": 4}
    drawn = subsets.draw(synthetic(), seed=1, size=120)
    assert drawn.strata == {"multi-session": 32, "temporal-reasoning": 32, "knowledge-update": 19,
                            "single-session-user": 17, "single-session-assistant": 13,
                            "single-session-preference": 7}
    assert all(sum(1 for qid in drawn.ids if qid.startswith(kind + "-")) == count
               for kind, count in drawn.strata.items())


def test_quotas_and_an_excluded_subset_keep_two_subsets_apart():
    subsets = _module("subsets")
    dev = subsets.draw(synthetic(), seed=3, quotas={"knowledge-update": 4, "single-session-user": 4,
                                                    "temporal-reasoning": 2, "multi-session": 2})
    assert dev.strata == {"knowledge-update": 4, "single-session-user": 4, "temporal-reasoning": 2,
                          "multi-session": 2}
    reported = subsets.draw(synthetic(), seed=3, size=60, exclude=dev.ids)
    assert not set(reported.ids) & set(dev.ids) and reported.excluded == tuple(sorted(dev.ids))
    assert len(reported.ids) == 60


def test_strata_can_split_abstention_questions_from_the_rest():
    subsets = _module("subsets")
    drawn = subsets.draw(synthetic({"multi-session": 20}, abstention_every=4), seed=1, size=10,
                         by="type+abstention")
    assert drawn.strata == {"multi-session": 8, "multi-session/abstention": 2}   # a tie goes to the larger


def test_a_recorded_subset_is_checked_against_the_data_it_came_from():
    subsets, records = _module("subsets"), _module("records")
    record = json.loads(json.dumps(subsets.draw(synthetic(), seed=5, size=60, exclude=("multi-session-1",)).record()))
    assert record["dataset"] == "synthetic" and record["seed"] == "5" and record["size"] == 60
    assert subsets.check(record, synthetic()).ids == tuple(record["ids"])
    with pytest.raises(records.DatasetError, match="SHA-256"):
        subsets.check(record, synthetic(sha="1" * 64))
    tampered = dict(record, ids=[*record["ids"][1:], "multi-session-1"])
    with pytest.raises(records.DatasetError, match="ids"):
        subsets.check(tampered, synthetic())
    for size in (500, 59, "60", None):
        with pytest.raises(records.DatasetError, match="size"):
            subsets.check(dict(record, size=size), synthetic())
    with pytest.raises(records.DatasetError, match="size"):
        subsets.check({key: value for key, value in record.items() if key != "size"}, synthetic())
    with pytest.raises(records.DatasetError, match="ids"):   # consistent, but not the size its seed draws
        subsets.check(dict(record, size=59, ids=record["ids"][1:]), synthetic())


def test_a_quota_past_its_stratum_or_a_size_past_the_data_is_refused():
    subsets, records = _module("subsets"), _module("records")
    with pytest.raises(records.DatasetError, match="single-session-preference"):
        subsets.draw(synthetic(), seed=1, quotas={"single-session-preference": 31})
    with pytest.raises(records.DatasetError, match="no-such-type"):
        subsets.draw(synthetic(), seed=1, quotas={"no-such-type": 1})
    with pytest.raises(records.DatasetError, match="501"):
        subsets.draw(synthetic(), seed=1, size=501)
    with pytest.raises(records.DatasetError, match="size or quotas"):
        subsets.draw(synthetic(), seed=1, size=10, quotas={"multi-session": 1})


def test_a_negative_size_or_quota_is_refused_rather_than_read_from_the_end():
    subsets, records = _module("subsets"), _module("records")
    with pytest.raises(records.DatasetError, match="-60"):
        subsets.draw(synthetic(), seed=1, size=-60)
    with pytest.raises(records.DatasetError, match="-5 multi-session"):
        subsets.draw(synthetic(), seed=1, quotas={"multi-session": -5})
    with pytest.raises(records.DatasetError, match="whole number"):
        subsets.draw(synthetic(), seed=1, quotas={"multi-session": 1.5})
    record = subsets.draw(synthetic(), seed=1, quotas={"multi-session": 3}).record()
    with pytest.raises(records.DatasetError, match="-3 multi-session"):
        subsets.check(dict(record, strata={"multi-session": -3}), synthetic())



# -- the command ---------------------------------------------------------------------------------------------------


@pytest.fixture
def small_longmemeval(tmp_path, monkeypatch):
    """The LongMemEval adapter pinned to a small file in ``tmp_path`` with its own counts."""
    lme = _module("longmemeval_s")
    path = lme_file(tmp_path, [
        lme_item(f"q{number}", "multi-session" if number % 2 else "knowledge-update", "2023-05-22 10:00",
                 [("s", "2023-05-20 08:00", chat(True))], ["s"]) for number in range(10)])
    pin = _pin(path.name, path.read_bytes())
    monkeypatch.setattr(lme, "PIN", pin)
    monkeypatch.setattr(lme, "PINNED", (pin,))
    monkeypatch.setattr(lme, "EXPECTED_TYPES", {"multi-session": 5, "knowledge-update": 5})
    monkeypatch.setattr(lme, "EXPECTED_ABSTENTION", 0)
    return tmp_path


def test_every_adapter_lists_the_files_it_pins():
    lme, beam, pmb = (_module(name) for name in ("longmemeval_s", "beam", "precisionmembench"))
    assert lme.PINNED == (lme.PIN, lme.JUDGE_PIN) and beam.PINNED == (beam.PIN,)
    assert pmb.PINNED == tuple(pmb.PINS.values())
    for module in (lme, beam, pmb):
        for pin in module.PINNED:
            assert len(pin.sha256) == 64 and pin.url.startswith("https://") and pin.size > 0


def test_the_command_verifies_a_dataset_and_reports_its_counts_and_findings(small_longmemeval, capsys):
    command = _module("__main__")
    assert command.main(["verify", "longmemeval_s", "--data-dir", str(small_longmemeval), "--offline"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts"] == {"knowledge-update": 5, "multi-session": 5} and report["questions"] == 10
    assert report["findings"]["reordered"] == 0 and report["sha256"] == _module("longmemeval_s").PIN.sha256


def test_the_command_draws_a_subset_into_a_record_and_checks_it(small_longmemeval, capsys):
    command = _module("__main__")
    out = small_longmemeval / "subset.json"
    assert command.main(["subset", "longmemeval_s", "--seed", "4", "--size", "4", "--out", str(out),
                         "--data-dir", str(small_longmemeval), "--offline"]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["strata"] == {"multi-session": 2, "knowledge-update": 2} and len(record["ids"]) == 4
    later = small_longmemeval / "later.json"
    assert command.main(["subset", "longmemeval_s", "--seed", "4", "--quota", "multi-session=3",
                         "--exclude", str(out), "--out", str(later), "--data-dir", str(small_longmemeval),
                         "--offline"]) == 0
    assert not set(json.loads(later.read_text(encoding="utf-8"))["ids"]) & set(record["ids"])
    assert command.main(["check", "longmemeval_s", str(out), "--data-dir", str(small_longmemeval),
                         "--offline"]) == 0
    record["ids"][0] = "q-not-drawn"
    out.write_text(json.dumps(record), encoding="utf-8")
    capsys.readouterr()
    assert command.main(["check", "longmemeval_s", str(out), "--data-dir", str(small_longmemeval),
                         "--offline"]) == 1
    assert "ids" in capsys.readouterr().err


def test_the_command_refuses_a_negative_size_and_writes_no_record(small_longmemeval, capsys):
    command = _module("__main__")
    out = small_longmemeval / "negative.json"
    assert command.main(["subset", "longmemeval_s", "--seed", "4", "--size=-4", "--out", str(out),
                         "--data-dir", str(small_longmemeval), "--offline"]) == 1
    assert "-4" in capsys.readouterr().err and not out.exists()


def test_the_command_refuses_a_record_whose_size_is_not_its_ids(small_longmemeval, capsys):
    command = _module("__main__")
    out = small_longmemeval / "subset.json"
    assert command.main(["subset", "longmemeval_s", "--seed", "4", "--size", "4", "--out", str(out),
                         "--data-dir", str(small_longmemeval), "--offline"]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    out.write_text(json.dumps(dict(record, size=10)), encoding="utf-8")
    capsys.readouterr()
    assert command.main(["check", "longmemeval_s", str(out), "--data-dir", str(small_longmemeval),
                         "--offline"]) == 1
    captured = capsys.readouterr()
    assert "size" in captured.err and "matches" not in captured.out


def test_the_command_refuses_a_missing_file_offline_and_says_how_to_fetch_it(tmp_path, capsys):
    command = _module("__main__")
    assert command.main(["verify", "beam", "--data-dir", str(tmp_path), "--offline"]) == 1
    assert "fetch" in capsys.readouterr().err


def test_the_fetch_command_offline_checks_the_cache_and_never_downloads(small_longmemeval, tmp_path, monkeypatch,
                                                                         capsys):
    command, pinned = _module("__main__"), _module("pinned")
    calls = []
    monkeypatch.setattr(pinned, "fetch", functools.partial(pinned.fetch, opener=_opener(b"never read", calls)))
    assert command.main(["fetch", "longmemeval_s", "--data-dir", str(small_longmemeval), "--offline"]) == 0
    assert _module("longmemeval_s").PIN.sha256 in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert command.main(["fetch", "beam", "--data-dir", str(empty), "--offline"]) == 1
    assert "fetch" in capsys.readouterr().err
    assert calls == [] and sorted(empty.iterdir()) == []


# -- the pinned upstream files, when the cache holds them ----------------------------------------------------------


def _pinned_directory(*pins) -> Path:
    directory = os.environ.get("COLONYMIND_BENCH_DATA_DIR")
    if not directory or not all((Path(directory) / pin.name).exists() for pin in pins):
        pytest.skip("the pinned upstream files are not in COLONYMIND_BENCH_DATA_DIR")
    return Path(directory)


@pytest.fixture(scope="module")
def pinned_longmemeval():
    lme = _module("longmemeval_s")
    return lme.load(_pinned_directory(lme.PIN), download=False)


def test_the_pinned_longmemeval_file_parses_to_the_published_counts(pinned_longmemeval):
    lme, dataset = _module("longmemeval_s"), pinned_longmemeval
    assert dataset.sha256 == lme.PIN.sha256
    assert dataset.counts() == lme.EXPECTED_TYPES and len(dataset.abstention_ids()) == lme.EXPECTED_ABSTENTION
    assert {name: len(ids) for name, ids in dataset.findings.items()} == {
        "reordered": 211, "tied_sessions": 95, "asked_after_last_session": 76, "repeated_sessions": 13,
        "unmarked_answer_sessions": 62}
    assert set(dataset.abstention_ids()) <= set(dataset.findings["unmarked_answer_sessions"])
    assert sum(len(item.evidence) for item in dataset.questions) == 896
    for item in dataset.questions:
        sessions = dataset.conversations[item.conversation].sessions
        assert all(earlier.at <= later.at for earlier, later in zip(sessions, sessions[1:]))
        assert all(session.at <= item.asked_at for session in sessions)
        assert item.asked_at.date() == lme.parse_stamp(item.date_text, item.id).date()


def test_the_pinned_beam_file_parses_to_the_published_counts_with_turn_evidence():
    beam = _module("beam")
    dataset = beam.load(_pinned_directory(beam.PIN), download=False)
    assert dataset.sha256 == beam.PIN.sha256 and beam.LICENSE == "CC BY-SA 4.0"
    assert dataset.counts() == beam.EXPECTED_TYPES and len(dataset.conversations) == beam.EXPECTED_CONVERSATIONS
    assert len(dataset.abstention_ids()) == 40
    sessions = [session for conversation in dataset.conversations.values() for session in conversation.sessions]
    assert len(sessions) == 90
    assert sum(1 for session in sessions for turn in session.turns if turn.role == "user") == 2866
    assert {name: len(ids) for name, ids in dataset.findings.items()} == {"reordered": 0, "no_turn_evidence": 5}
    assert sum(1 for item in dataset.questions if item.evidence) == 355
    for item in dataset.questions:
        conversation = dataset.conversations[item.conversation].sessions
        assert all(session.at < item.asked_at for session in conversation)



def test_the_pinned_precisionmembench_fixtures_parse_to_their_counts():
    pmb = _module("precisionmembench")
    corpus = pmb.load(_pinned_directory(*pmb.PINS.values()), download=False)
    assert corpus.counts() == pmb.EXPECTED_CATEGORIES and len(corpus.beliefs) == 35
    assert len(corpus.sessions) == 2 and corpus.scored_turns() == 12
    assert [item.id for item in corpus.beliefs if item.user != "test-user"] == ["b-other-user-leak-check"]
    assert sum(1 for item in corpus.beliefs if item.superseded_by) == 4
    assert corpus.findings == {"unread_conditions": ("session-drift-noise-10-turns-implicit-redis:10",)}


def test_a_subset_of_the_pinned_longmemeval_file_is_recorded_and_checks_against_it(pinned_longmemeval):
    subsets = _module("subsets")
    dev = subsets.draw(pinned_longmemeval, seed=12, quotas={"knowledge-update": 4, "single-session-user": 4,
                                                            "temporal-reasoning": 2, "multi-session": 2})
    drawn = subsets.draw(pinned_longmemeval, seed=60, size=60, exclude=dev.ids)
    assert drawn.strata == {"multi-session": 16, "temporal-reasoning": 16, "knowledge-update": 9,
                            "single-session-user": 8, "single-session-assistant": 7, "single-session-preference": 4}
    record = json.loads(json.dumps(drawn.record()))
    assert record["sha256"] == _module("longmemeval_s").PIN.sha256 and not set(record["ids"]) & set(dev.ids)
    assert subsets.check(record, pinned_longmemeval) == drawn
