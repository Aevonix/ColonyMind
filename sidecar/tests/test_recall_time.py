"""Recall and time: question dates resolved as spans, the date-window lane, and a search's own window and reference
time (docs/RECALL-HYBRID.md, "Recall and time"; docs/SOURCE-CLAIMS.md, "Recall and time").

"What did we talk about last week" names a span of time, not words: its messages share no word with the question and
often nothing a vector finds. The question's date is resolved against the turn's own time as a span, and a date-window
lane reads the person's sources in that span straight from the ledger, spread across the span, so the reranker can
judge them. An explicit search can also name its window and the time its relative dates count from.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind import recall_cli
from colonymind.beliefs.source_time import EARLIEST, MemoryTimeQuery, filter_unstructured, interpret_time_query
from colonymind.api.schemas.host import RecallTraceReport
from colonymind.memory import trace as recall_trace
from colonymind.memory.recall import source_candidates
from colonymind.memory import search
from colonymind.memory.search import CollectedSources, collect_sources, select_memory
from colonymind.memory.selection import RecallSelector
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import source_message_hash
from colonymind.turns.source_vectors import SourceVectors
from onekey import KEY
from colonymind.api.middleware import ApiKeyMiddleware
from test_turn_source_evidence import source_app  # noqa: F401 (fixture)


# --- The trace and the recall command ------------------------------------------------------------------------------

def _traced_packet(tmp_path, *, limit):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("t-1", contact_id="p", session_id="s", derive_claims=False,
                         messages=[{"role": "user", "content": "The kiln reached cone six."}])
    rows = source_candidates(ledger.search_sources("kiln", contact_id="p", session_id="s"))
    tracer = recall_trace.RecallTrace()
    tracer.lane("stemmed", ledger.search_stems("kiln", contact_id="p", session_id="s"))
    tracer.recall(CollectedSources(ledger, "p", "s", None), limit=limit, max_chars=4000)
    tracer.fused(rows)
    tracer.packing(rows, limit=limit, max_chars=4000)
    tracer.packet(rows, "packet")
    return tracer.report()


def test_a_packet_with_no_count_limit_reports_none(tmp_path):
    report = _traced_packet(tmp_path, limit=None)
    assert report["packet"] == {"limit": None, "max_chars": 4000, "count": 1}
    assert RecallTraceReport.model_validate(report).packet.limit is None
    assert _traced_packet(tmp_path / "limited", limit=5)["packet"]["limit"] == 5
    first = recall_cli.explain({"count": 1, "trace": report}, person="p")[0]
    assert first == "Recall for p: 1 record in the packet (no count limit), within 4000 characters."


def test_the_recall_command_names_every_lane_in_words():
    row = {"turn_id": "t-1", "candidate": "source-excerpt:" + "a" * 64, "form": "quotation", "fused_rank": 1,
           "lanes": {"lexical": 2, "stemmed": 1, "window": 3, "claim_key": 4}, "exit": None}
    trace = {"lanes": {"lexical": 1, "stemmed": 1, "window": 1, "claim_key": 1}, "rows": [row],
             "packet": {"limit": 5, "max_chars": 4000, "count": 1}, "calibration": {"status": "none"}}
    lines = recall_cli.explain({"count": 1, "trace": trace}, person="p")
    assert lines[0] == "Recall for p: 1 of at most 5 records in the packet, within 4000 characters."
    assert lines[1] == "Lanes: keyword found 1, stemmed keyword found 1, date window found 1, claim key found 1."
    assert ("found by keyword (rank 2) and stemmed keyword (rank 1) and date window (rank 3) and claim key (rank 4)"
            in lines[3])
    assert "_" not in lines[1] + lines[3].split(": ", 1)[1]


def test_the_recall_commands_limit_says_whose_default_it_is():
    import argparse
    parser = argparse.ArgumentParser()
    recall_cli.add_parser(parser.add_subparsers(dest="command"))
    action, = [action for action in parser._subparsers._group_actions[0].choices["recall"]._actions
               if action.dest == "limit"]
    assert "default 5, a turn's" not in action.help and "default 5, an explicit search's" in action.help
    assert action.default == 5


# --- Question dates as spans ---------------------------------------------------------------------------------------

UTC = timezone.utc
#: A Wednesday, 10:00 in New York and already Thursday 00:00 in Tokyo: the week, the month and the year all turn
#: over within the spans below.
NOW = datetime(2027, 1, 6, 15, tzinfo=UTC)


def z(stamp: str) -> str:
    return datetime.fromisoformat(stamp).astimezone(UTC).isoformat()


# (question, zone, start, end): local calendar spans, as UTC instants, end exclusive.
SPANS = [
    ("What did we talk about last week?", "America/New_York", "2026-12-28T05:00:00", "2027-01-04T05:00:00"),
    ("What did we talk about last week?", "Asia/Tokyo", "2026-12-27T15:00:00", "2027-01-03T15:00:00"),
    ("What did I say this week?", "America/New_York", "2027-01-04T05:00:00", "2027-01-11T05:00:00"),
    ("What did I say this week?", "Asia/Tokyo", "2027-01-03T15:00:00", "2027-01-10T15:00:00"),
    ("What did I plan last month?", "America/New_York", "2026-12-01T05:00:00", "2027-01-01T05:00:00"),
    ("What did I plan last month?", "Asia/Tokyo", "2026-11-30T15:00:00", "2026-12-31T15:00:00"),
    ("What did I read last year?", "America/New_York", "2026-01-01T05:00:00", "2027-01-01T05:00:00"),
    ("What did I read last year?", "Asia/Tokyo", "2025-12-31T15:00:00", "2026-12-31T15:00:00"),
    # The latest March that has begun; daylight saving starts inside it in New York.
    ("What did I cook in March?", "America/New_York", "2026-03-01T05:00:00", "2026-04-01T04:00:00"),
    ("What did I cook in March?", "Asia/Tokyo", "2026-02-28T15:00:00", "2026-03-31T15:00:00"),
    ("What did I cook in March 2025?", "UTC", "2025-03-01T00:00:00", "2025-04-01T00:00:00"),
    ("What did I read during 2024?", "UTC", "2024-01-01T00:00:00", "2025-01-01T00:00:00"),
    # The last weekend that has begun: Saturday and Sunday.
    ("What did I do over the weekend?", "America/New_York", "2027-01-02T05:00:00", "2027-01-04T05:00:00"),
    ("What did I do over the weekend?", "Asia/Tokyo", "2027-01-01T15:00:00", "2027-01-03T15:00:00"),
    ("What did I say 3 days ago?", "America/New_York", "2027-01-03T05:00:00", "2027-01-04T05:00:00"),
    ("What did I say 3 days ago?", "Asia/Tokyo", "2027-01-03T15:00:00", "2027-01-04T15:00:00"),
    # The calendar week that held the day two weeks back.
    ("What did I say two weeks ago?", "America/New_York", "2026-12-21T05:00:00", "2026-12-28T05:00:00"),
    ("What did I say two weeks ago?", "Asia/Tokyo", "2026-12-20T15:00:00", "2026-12-27T15:00:00"),
    # Daylight saving ends on the first of November in New York, after midnight.
    ("What did I say 2 months ago?", "America/New_York", "2026-11-01T04:00:00", "2026-12-01T05:00:00"),
    ("What did I say 2 months ago?", "Asia/Tokyo", "2026-10-31T15:00:00", "2026-11-30T15:00:00"),
    ("What did I say a year ago?", "UTC", "2026-01-01T00:00:00", "2027-01-01T00:00:00"),
    # Seasons follow the hemisphere the time zone lies in.
    ("Where did I go last summer?", "America/New_York", "2026-06-01T04:00:00", "2026-09-01T04:00:00"),
    ("Where did I go last summer?", "Australia/Sydney", "2025-11-30T13:00:00", "2026-02-28T13:00:00"),
    ("What did we discuss between December 20, 2026 and December 24, 2026?", "America/New_York",
     "2026-12-20T05:00:00", "2026-12-25T05:00:00"),
    ("What did we discuss between last month and this week?", "UTC", "2026-12-01T00:00:00", "2027-01-11T00:00:00"),
]


@pytest.mark.parametrize("question,zone,start,end", SPANS)
def test_a_relative_date_resolves_to_its_span_against_the_turns_time(question, zone, start, end):
    query = interpret_time_query(question, now=NOW, timezone_name=zone)
    assert query.mode == "valid_range"
    assert (query.start, query.end) == (z(start + "+00:00"), z(end + "+00:00"))
    assert query.window() == (datetime.fromisoformat(query.start), datetime.fromisoformat(query.end))
    assert not query.supplied


def test_a_month_boundary_and_the_events_mode():
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)  # a Thursday
    ago = interpret_time_query("What happened 2 days ago?", now=now)
    assert (ago.mode, ago.start, ago.end) == ("observed_range", "2026-09-29T00:00:00+00:00",
                                              "2026-09-30T00:00:00+00:00")
    week = interpret_time_query("What happened last week?", now=now)
    assert (week.start, week.end) == ("2026-09-21T00:00:00+00:00", "2026-09-28T00:00:00+00:00")
    month = interpret_time_query("What did I say this month?", now=now)
    assert (month.start, month.end) == ("2026-10-01T00:00:00+00:00", "2026-11-01T00:00:00+00:00")
    last = interpret_time_query("What did I say last month?", now=now)
    assert (last.start, last.end) == ("2026-09-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00")
    # On a Saturday the weekend under way is "the weekend"; "last weekend" is the one before it.
    saturday = datetime(2026, 10, 3, 12, tzinfo=UTC)
    assert interpret_time_query("What did I do over the weekend?", now=saturday).start == "2026-10-03T00:00:00+00:00"
    assert interpret_time_query("What did I do last weekend?", now=saturday).start == "2026-09-26T00:00:00+00:00"


def test_since_before_and_between_bound_the_span():
    since = interpret_time_query("What was recorded since last week?", now=NOW, timezone_name="America/New_York")
    assert (since.mode, since.start, since.end) == ("observed_range", "2026-12-28T05:00:00+00:00", NOW.isoformat())
    before = interpret_time_query("Where was my office before December 20, 2026?", now=NOW,
                                  timezone_name="America/New_York")
    assert (before.mode, before.start, before.end) == ("valid_range", EARLIEST, "2026-12-20T05:00:00+00:00")
    assert before.accepts_claim({"valid_from": "2020-01-01T00:00:00+00:00", "valid_to": "2021-01-01T00:00:00+00:00"})
    assert not before.accepts_claim({"valid_from": "2026-12-21T00:00:00+00:00"})
    seen = interpret_time_query("What was seen before 2026-12-20?", now=NOW)
    assert (seen.mode, seen.start, seen.end) == ("observed_range", EARLIEST, "2026-12-20T00:00:00+00:00")
    assert seen.accepts_observation("2019-05-01T00:00:00+00:00") and not seen.accepts_observation(NOW.isoformat())


@pytest.mark.parametrize("question", [
    "Where is my office next week?",
    "Where was my office after March 3, 2026?",
    "Where was my office until 2026-03-12?",
    "Have I told you this before?",
    "Where was my office before I moved?",
    "Where was my office between March 1, 2026 and the move?",
    "Where was my office between March 5, 2026 and March 1, 2026?",
    "Where was my office on 1 March 2026 or 5 March 2026?",
    "Where was my office last week or last month?",
    "Where was my office the previous week?",
    # A count the parser does not read whole is not read in part: a decimal, a thousands separator, four digits.
    "What did I say 1.5 months ago?",
    "What did I say 1,000 months ago?",
    "What did I say 1000 weeks ago?",
    "What did I say 2.5 days ago?",
    'What did I say "1.5 months ago"?',
])
def test_unresolved_forms_still_admit_no_claims(question):
    query = interpret_time_query(question, now=NOW)
    assert query.mode == "unresolved_time" and query.window() is None
    assert not query.accepts_claim({"valid_from": "2026-01-01T00:00:00+00:00"})
    assert not query.accepts_claim({"event_at": "2026-12-30T00:00:00+00:00"})


def test_a_season_the_time_zone_cannot_place_stays_unparsed():
    # "Last summer" means different months either side of the equator: without a hemisphere it is not read.
    for zone in ("UTC", "Asia/Singapore"):
        query = interpret_time_query("Where did I go last summer?", now=NOW, timezone_name=zone)
        assert query.mode == "current" and query.window() is None


def test_a_quoted_relative_date_is_read_as_the_same_operand():
    quoted = interpret_time_query('What did I say "last week"?', now=NOW)
    plain = interpret_time_query("What did I say last week?", now=NOW)
    assert (quoted.start, quoted.end) == (plain.start, plain.end) == (
        "2026-12-28T00:00:00+00:00", "2027-01-04T00:00:00+00:00")
    # A date inside a pasted report is evidence, not the question's time.
    assert interpret_time_query('Inspect this report: "It rained last week."', now=NOW).mode == "current"


def test_a_supplied_window_sets_the_claim_view_and_keeps_every_quotation():
    window = (datetime(2026, 12, 1, tzinfo=UTC), datetime(2026, 12, 8, tzinfo=UTC))
    supplied = interpret_time_query("What did the camera record?", now=NOW, window=window)
    assert (supplied.mode, supplied.start, supplied.end, supplied.supplied) == (
        "observed_range", "2026-12-01T00:00:00+00:00", "2026-12-08T00:00:00+00:00", True)
    assert supplied.accepts_claim({"event_at": "2026-12-02T00:00:00+00:00"})
    assert not supplied.accepts_claim({"event_at": "2026-12-09T00:00:00+00:00"})
    # The supplied window wins over a date in the words.
    assert interpret_time_query("What did the camera record last year?", now=NOW, window=window).start == (
        "2026-12-01T00:00:00+00:00")
    inside = {"id": "a", "occurred_at": "2026-12-02T00:00:00+00:00"}
    outside = {"id": "b", "occurred_at": "2026-11-02T00:00:00+00:00"}
    undated = {"id": "c"}
    assert [(row["id"], row["validity_status"]) for row in filter_unstructured([inside, outside, undated],
                                                                                supplied)] == [
        ("a", "source_occurrence_only"), ("b", "outside_requested_window"), ("c", "unknown")]
    # The same window read from the words drops what falls outside it, as before.
    parsed = MemoryTimeQuery("observed_range", supplied.start, supplied.end)
    assert [row["id"] for row in filter_unstructured([inside, outside, undated], parsed)] == ["a"]
    valid = interpret_time_query("Where was my office?", now=NOW, window=window)
    assert valid.mode == "valid_range" and valid.supplied
    assert [row["validity_status"] for row in filter_unstructured([inside, outside], valid)] == ["unknown"] * 2


def test_as_of_is_the_reference_for_relative_dates():
    as_of = NOW - timedelta(days=30)
    query = interpret_time_query("What did I say yesterday?", now=as_of)
    assert query.start == "2026-12-06T00:00:00+00:00"
    current = interpret_time_query("Where is my office?", now=as_of)
    assert current.mode == "current" and current.start == as_of.isoformat()


# --- The date-window lane ------------------------------------------------------------------------------------------

LAST_WEEK = "What did we talk about last week?"   # Monday 28 December 2026 to Monday 4 January 2027, in UTC
WEEK = (datetime(2026, 12, 28, tzinfo=UTC), datetime(2027, 1, 4, tzinfo=UTC))


def ledger_at(tmp_path):
    return TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")


def record(ledger, turn_id, content, at, *, contact="owner", session="s-1", scope="person", role="user"):
    ledger.record_source(turn_id, contact_id=contact, session_id=session, scope=scope, occurred_at=at,
                         messages=[{"role": role, "content": content}], derive_claims=False)


def window_lane(ledger, query=LAST_WEEK, *, now=NOW, window=None, session="s-1", vector=()):
    """The window lane's hits, as recall runs it."""
    lane, = [lane for lane in search.LANES if lane.name == "window"]
    time_query = interpret_time_query(query, now=now, timezone_name="UTC", window=window)
    import asyncio
    return asyncio.run(lane.search(search.LaneQuery(ledger, query, "owner", session, time_query=time_query,
                                                    found={"vector": list(vector)}))).hits


def test_the_window_lane_is_registered_to_run_with_the_questions_time():
    lane, = [lane for lane in search.LANES if lane.name == "window"]
    assert lane.timed and lane.optional and not lane.keyword
    assert [lane.name for lane in search.LANES][-1] == "window"


async def test_an_in_window_source_no_lane_ranks_is_found_through_the_window_lane(tmp_path, monkeypatch):
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK", raising=False)
    ledger = ledger_at(tmp_path)
    record(ledger, "kiln", "The kiln reached cone six before lunch.", "2026-12-29T10:00:00+00:00")
    record(ledger, "older", "The glaze test cracked in the cold.", "2026-12-10T10:00:00+00:00")
    record(ledger, "talk", "We should talk about the budget soon.", "2026-11-01T10:00:00+00:00")
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(ledger, query=LAST_WEEK, contact_id="owner", session_id="s-1", trace=tracer)
    assert {hit["turn_id"] for hit in collected.hits} == {"talk"}        # the words find only the other one
    packet = await select_memory(collected, query=LAST_WEEK, selector=RecallSelector(), timezone_name="UTC",
                                 now=NOW, trace=tracer)
    assert "cone six" in packet.content and "glaze" not in packet.content
    report = tracer.report()
    kiln, = [row for row in report["rows"] if row["turn_id"] == "kiln"]
    assert kiln["lanes"] == {"window": 1} and kiln["exit"] is None      # the trace shows the window lane's rank
    assert report["lanes"]["window"] == 1


async def test_a_question_without_a_date_runs_no_window_lane(tmp_path, monkeypatch):
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK", raising=False)
    ledger = ledger_at(tmp_path)
    record(ledger, "kiln", "The kiln reached cone six before lunch.", "2026-12-29T10:00:00+00:00")
    tracer = recall_trace.RecallTrace()
    collected = await collect_sources(ledger, query="What about the kiln?", contact_id="owner", session_id="s-1",
                                      trace=tracer)
    await select_memory(collected, query="What about the kiln?", selector=RecallSelector(), timezone_name="UTC",
                        now=NOW, trace=tracer)
    assert "window" not in tracer.report()["lanes"]


async def test_a_turn_gets_the_window_lane(source_app, tmp_path, monkeypatch):  # noqa: F811 (fixture)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    monkeypatch.setattr(SourceVectors, "search", AsyncMock(return_value=([], [])))
    ledger = ledger_at(tmp_path)
    now = datetime.now(UTC)
    record(ledger, "kiln", "The kiln reached cone six before lunch.", (now - timedelta(days=7)).isoformat(),
           contact="person")
    record(ledger, "older", "The glaze test cracked in the cold.", (now - timedelta(days=60)).isoformat(),
           contact="person")
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url="http://fixture") as client:
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "fixture"}, "context": {"contact_id": "person", "session_id": "later"},
            "incoming_message": {"role": "user", "content": LAST_WEEK}})
    assert response.status_code == 200, response.text
    memory, = [section for section in response.json()["sections"] if section["id"] == "colonymind-memory"]
    assert "cone six" in memory["body"] and "glaze" not in memory["body"]


def test_a_supplied_window_finds_the_same_in_window_sources_as_the_parsed_one(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "mon", "The kiln reached cone six.", "2026-12-28T09:00:00+00:00")
    record(ledger, "sun", "We talked about the harbour lights.", "2027-01-03T20:00:00+00:00")
    record(ledger, "before", "The glaze test cracked.", "2026-12-27T23:59:59+00:00")
    record(ledger, "after", "The ferry was late.", "2027-01-04T00:00:00+00:00")
    parsed = window_lane(ledger)
    supplied = window_lane(ledger, "What did we talk about?", window=WEEK)
    assert {hit["turn_id"] for hit in parsed} == {hit["turn_id"] for hit in supplied} == {"mon", "sun"}


def test_the_lane_orders_windowed_keywords_then_vector_ranks_then_newest(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "glaze", "The glaze came out a deep blue.", "2026-12-28T09:00:00+00:00")
    record(ledger, "vector", "A cobalt colour for the next firing.", "2026-12-28T10:00:00+00:00")
    record(ledger, "newest", "Lunch was late.", "2026-12-28T11:00:00+00:00")
    record(ledger, "old-glaze", "The glaze was green then.", "2026-11-28T09:00:00+00:00")
    vector = ledger.search_sources("cobalt colour", contact_id="owner", session_id="s-1")
    vector = [dict(hit, retrieval_method="semantic") for hit in vector]
    hits = window_lane(ledger, "What did I say about the glaze last week?", vector=vector)
    assert [hit["turn_id"] for hit in hits] == ["glaze", "vector", "newest"]
    assert hits[1] == vector[0]                                  # the vector lane's own hit, so the two fuse


def test_the_windowed_keyword_rank_looks_past_better_matches_outside_the_scope_or_span(tmp_path, monkeypatch):
    from colonymind.memory import time_lane
    monkeypatch.setattr(time_lane, "SCAN_LIMIT", 40)       # the first read of the best matches, made small
    ledger = ledger_at(tmp_path)
    for index in range(30):   # better matches than the in-span one: before the span, and another person's in it
        record(ledger, f"before-{index}", "Glaze, glaze and more glaze.", "2026-11-20T10:00:00+00:00")
        record(ledger, f"sam-{index}", "Glaze, glaze and more glaze.", "2026-12-30T10:00:00+00:00", contact="sam")
    record(ledger, "relevant", "The glaze came out a deep blue.", "2027-01-03T08:00:00+00:00")
    for index in range(20):   # newer in-span messages that do not match: newest first alone would fill the lane
        record(ledger, f"noise-{index:02d}", f"Harbour note {index}.", f"2027-01-03T{9 + index % 14:02d}:{index:02d}:00+00:00")
    turns = [hit["turn_id"] for hit in window_lane(ledger, "What did I say about the glaze last week?")]
    assert turns[0] == "relevant" and len(turns) == 15


def test_a_week_of_unmatched_messages_reaches_the_pool_spread_across_the_week(tmp_path):
    ledger = ledger_at(tmp_path)
    for hour in range(20):   # a busy Sunday: newest first alone would fill the lane with it
        record(ledger, f"sun-{hour:02d}", f"Harbour note {hour}.", f"2027-01-03T{hour:02d}:00:00+00:00")
    for day, name in enumerate(("mon", "tue", "wed", "thu")):
        record(ledger, name, f"Workshop note {name}.", f"2026-12-{28 + day}T12:00:00+00:00")
    hits = window_lane(ledger)
    turns = [hit["turn_id"] for hit in hits]
    assert len(turns) == 15 and len(set(turns)) == 15
    assert {"mon", "tue", "wed", "thu"} <= set(turns[:8])         # the first round takes one from each part
    assert turns[0] == "sun-19"                                   # the newest leads its part


def test_messages_found_through_their_claims_are_spread_across_the_span_by_event_date(tmp_path):
    # Every message is sent after the week, about a day in it: one about Monday, many about a busy Sunday (from 04:00,
    # the week's last eighth).
    ledger = ledger_at(tmp_path)
    told = "2027-01-06T09:00:00+00:00"
    record(ledger, "monday", "The kiln was fired.", told)
    claim(ledger, "monday", "c-monday", {"event_at": "2026-12-28T12:00:00+00:00", "event_time": {}})
    for index in range(64):
        record(ledger, f"sunday-{index:02d}", f"Harbour note {index}.", told)
        claim(ledger, f"sunday-{index:02d}", f"c-sunday-{index}", {
            "event_at": f"2027-01-03T{4 + index % 20:02d}:{index // 20:02d}:00+00:00", "event_time": {}})
    turns = [hit["turn_id"] for hit in window_lane(ledger)]
    assert len(turns) == 15 and "monday" in turns[:2]   # the first round takes one from each part the events fill


def test_the_span_is_divided_as_asked_not_by_what_it_holds(tmp_path):
    # A week whose messages all fall on its weekend: its 8 parts are the week's, so the weekend fills two or three
    # of them, and the first round takes the newest of each of those before a second from any.
    ledger = ledger_at(tmp_path)
    for hour in range(0, 48, 3):
        at = datetime(2027, 1, 2, tzinfo=UTC) + timedelta(hours=hour)
        record(ledger, f"weekend-{hour:02d}", f"Weekend note {hour}.", at.isoformat())
    turns = [hit["turn_id"] for hit in window_lane(ledger)]
    width = (WEEK[1] - WEEK[0]) / 8
    parts = [int((datetime(2027, 1, 2, tzinfo=UTC) + timedelta(hours=int(turn.split("-")[1])) - WEEK[0]) / width)
             for turn in turns]
    assert len(set(parts)) == 3 and len(set(parts[:3])) == 3


def test_a_window_from_the_first_centuries_is_read_like_any_other(tmp_path):
    ledger = ledger_at(tmp_path)
    record(ledger, "kiln", "The kiln reached cone six.", "2026-12-29T10:00:00+00:00")
    for start in (datetime(1, 1, 1, tzinfo=UTC), datetime(1, 6, 1, tzinfo=UTC)):
        window = (start, datetime(2027, 1, 1, tzinfo=UTC))
        assert [hit["turn_id"] for hit in window_lane(ledger, "What did we say?", window=window)] == ["kiln"]


def test_the_window_lane_keeps_the_persons_scope(tmp_path):
    ledger = ledger_at(tmp_path)
    at = "2026-12-29T10:00:00+00:00"
    record(ledger, "mine", "Mine.", at)
    record(ledger, "theirs", "Theirs.", at, contact="sam")
    record(ledger, "this-session", "This session.", at, scope="session")
    record(ledger, "other-session", "Other session.", at, scope="session", session="s-9")
    assert {hit["turn_id"] for hit in window_lane(ledger)} == {"mine", "this-session"}
    assert {hit["turn_id"] for hit in window_lane(ledger, session="s-9")} == {"mine", "other-session"}


def claim(ledger, turn, claim_id, data, retracted=None):
    with closing(ledger._connect()) as conn, conn:
        session, messages = conn.execute("SELECT session_id, messages_json FROM turn_sources WHERE turn_id=?",
                                         (turn,)).fetchone()
        digest = source_message_hash(session, json.loads(messages)[0])
        conn.execute("INSERT INTO source_claims(id, turn_id, message_hash, subject_key, predicate, value_key, "
                     "data_json, retracted_by) VALUES (?, ?, ?, 'speaker', 'did', ?, ?, ?)",
                     (claim_id, turn, digest, claim_id, json.dumps(data), retracted))


def test_a_source_whose_claim_happened_in_the_window_is_in_the_lane(tmp_path):
    ledger = ledger_at(tmp_path)
    tuesday = {"status": "resolved", "precision": "calendar_day", "start": "2026-12-29T00:00:00+00:00",
               "end_exclusive": "2026-12-30T00:00:00+00:00", "expression": "Tuesday"}
    record(ledger, "told-today", "On Tuesday I fixed the gate.", "2027-01-06T09:00:00+00:00")
    claim(ledger, "told-today", "c-1", {"event_time": tuesday})
    record(ledger, "retracted", "On Tuesday I painted the shed.", "2027-01-06T09:05:00+00:00")
    claim(ledger, "retracted", "c-2", {"event_time": tuesday}, retracted="c-9")
    # A period that began weeks before the span and reaches into it; an instant inside it.
    record(ledger, "december", "In December I read three books.", "2027-01-05T09:00:00+00:00")
    claim(ledger, "december", "c-3", {"event_time": {"status": "resolved", "precision": "period", "period": "month",
                                                     "start": "2026-12-01T00:00:00+00:00",
                                                     "end_exclusive": "2027-01-01T00:00:00+00:00"}})
    record(ledger, "instant", "The alarm went off at noon on the 30th.", "2027-01-06T08:00:00+00:00")
    claim(ledger, "instant", "c-4", {"event_at": "2026-12-30T12:00:00+00:00", "event_time": {"status": "unknown"}})
    record(ledger, "outside", "In November I read one book.", "2027-01-05T10:00:00+00:00")
    claim(ledger, "outside", "c-5", {"event_time": {"status": "resolved", "precision": "period", "period": "month",
                                                    "start": "2026-11-01T00:00:00+00:00",
                                                    "end_exclusive": "2026-12-01T00:00:00+00:00"}})
    assert {hit["turn_id"] for hit in window_lane(ledger)} == {"told-today", "december", "instant"}
    from colonymind.memory import time_lane
    with closing(ledger._connect()) as conn:
        for sql, args in ((time_lane.CLAIM_INTERVAL_SQL, ("a", "b", "owner", "s-1", "a")),
                          (time_lane.CLAIM_POINT_SQL, ("a", "b", "owner", "s-1"))):
            plan = " ".join(row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, args))
            # The event-time index is read in order, newest first, so the reader can stop early.
            assert any(name in plan for name in time_lane.CLAIM_INDEXES) and "TEMP B-TREE" not in plan


def many_claims(ledger, turn, prefix, count, data, retracted=None):
    """``count`` claims on ``turn``'s message, each with ``data(index)``."""
    with closing(ledger._connect()) as conn, conn:
        session, messages = conn.execute("SELECT session_id, messages_json FROM turn_sources WHERE turn_id=?",
                                         (turn,)).fetchone()
        digest = source_message_hash(session, json.loads(messages)[0])
        conn.executemany("INSERT INTO source_claims(id, turn_id, message_hash, subject_key, predicate, value_key, "
                         "data_json, retracted_by) VALUES (?, ?, ?, 'speaker', 'did', ?, ?, ?)",
                         [(f"{prefix}-{index}", turn, digest, f"{prefix}-{index}", json.dumps(data(index)), retracted)
                          for index in range(count)])


def test_claims_the_person_cannot_see_never_crowd_out_their_own(tmp_path):
    # Newer events than the person's own, more of each kind than the lane once examined in all: another person's,
    # another session's, retracted ones, and periods that end before the span.
    ledger = ledger_at(tmp_path)
    told = "2027-01-06T09:00:00+00:00"
    record(ledger, "mine-period", "In December I read three books.", told)
    claim(ledger, "mine-period", "c-period", {"event_time": {
        "status": "resolved", "precision": "period", "period": "month", "start": "2026-12-01T00:00:00+00:00",
        "end_exclusive": "2027-01-01T00:00:00+00:00"}})
    record(ledger, "mine-instant", "The alarm went off at noon on the 28th.", told)
    claim(ledger, "mine-instant", "c-instant", {"event_at": "2026-12-28T12:00:00+00:00", "event_time": {}})
    record(ledger, "sam", "Sam's week.", told, contact="sam")
    record(ledger, "elsewhere", "Another session's notes.", told, scope="session", session="s-9")
    record(ledger, "retracted", "Notes taken back.", told)
    record(ledger, "early", "Early December notes.", told)

    def day(index):
        return f"2027-01-0{1 + index % 3}T{index % 24:02d}:00:00+00:00"
    for turn, retracted in (("sam", None), ("elsewhere", None), ("retracted", "c-9")):
        many_claims(ledger, turn, turn + "-instant", 1100, lambda index: {"event_at": day(index), "event_time": {}},
                    retracted)
        many_claims(ledger, turn, turn + "-period", 1100, lambda index: {"event_time": {
            "status": "resolved", "precision": "calendar_day", "start": day(index),
            "end_exclusive": "2027-01-04T00:00:00+00:00"}}, retracted)
    many_claims(ledger, "early", "early", 2100, lambda index: {"event_time": {
        "status": "resolved", "precision": "calendar_day", "start": "2026-12-10T00:00:00+00:00",
        "end_exclusive": "2026-12-11T00:00:00+00:00"}})
    assert {hit["turn_id"] for hit in window_lane(ledger)} == {"mine-period", "mine-instant"}


def test_the_ledger_builds_the_occurrence_index_the_lane_reads(tmp_path):
    from colonymind.memory import time_lane
    ledger = ledger_at(tmp_path)
    with closing(ledger._connect()) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                            (time_lane.INDEX,)).fetchone()
        plan = " ".join(row[-1] for row in conn.execute(
            "EXPLAIN QUERY PLAN " + time_lane.NEWEST_SQL, ("owner", "s-1", "2026-12-28", "2027-01-04", 15)))
    assert time_lane.INDEX in plan


def test_a_failed_window_lane_leaves_the_other_evidence(tmp_path, monkeypatch):
    import asyncio
    from colonymind.memory import time_lane
    ledger = ledger_at(tmp_path)
    record(ledger, "talk", "We should talk about the budget soon.", "2026-11-01T10:00:00+00:00")

    def broken(*args, **kwargs):
        raise RuntimeError("window read failed")
    monkeypatch.setattr(time_lane, "window_hits", broken)

    async def recall():
        collected = await collect_sources(ledger, query=LAST_WEEK, contact_id="owner", session_id="s-1")
        return await select_memory(collected, query=LAST_WEEK, selector=RecallSelector(), timezone_name="UTC",
                                   now=NOW)
    assert "budget" in asyncio.run(recall()).content


# --- A supplied window never hides a projected source ------------------------------------------------------------

async def projected(ledger, turn, text, extracted, at):
    """A source whose claims the (controlled) extractor wrote, as ingestion leaves it."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from test_source_claim_projection import Model
    ledger.record_source(turn, contact_id="owner", session_id="s-1", occurred_at=at,
                         messages=[{"role": "user", "content": text}])
    assert await SourceClaimProjection(ledger).process_one(Model({text: extracted}))


async def recalled(ledger, question, *, window=None):
    collected = await collect_sources(ledger, query=question, contact_id="owner", session_id="s-1")
    time_query = interpret_time_query(question, now=NOW, timezone_name="UTC", window=window)
    return await select_memory(collected, query=question, selector=RecallSelector(), timezone_name="UTC", now=NOW,
                               time_query=time_query)


NOVEMBER = (datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 8, tzinfo=UTC))


@pytest.mark.parametrize("question,label", [
    ("What happened at the gate with the fox?", "outside_requested_window"),  # an event window: the event is dated
    ("Which fox was at the gate?", "unknown"),                                # a validity window: nothing dates it
])
async def test_a_supplied_window_keeps_a_source_its_claim_dates_outside_it(tmp_path, monkeypatch, question, label):
    from test_source_claim_projection import claim as extracted
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK", raising=False)
    ledger = ledger_at(tmp_path)
    text = "I saw a fox at the gate on 1 October 2026."
    # The claim covers the whole message, so the projection reads the message through it.
    await projected(ledger, "gate", text, extracted(text, "a fox at the gate", predicate="sighting",
                                                    event_at_text="1 October 2026"), "2026-10-01T18:00:00+00:00")
    assert "fox at the gate" in (await recalled(ledger, question)).content
    kept = await recalled(ledger, question, window=NOVEMBER)
    assert "fox at the gate" in kept.content and f'"validity_status": "{label}"' in kept.content
    # The same window read from the words keeps today's rule for an event window: what falls outside it is left out.
    if "happened" in question:
        assert "fox at the gate" not in (await recalled(ledger, question + " In November 2026?")).content


async def test_a_supplied_window_never_brings_back_a_corrected_quotation(tmp_path, monkeypatch):
    from test_source_claim_projection import claim as extracted
    monkeypatch.delenv("COLONYMIND_RECALL_RERANK", raising=False)
    ledger = ledger_at(tmp_path)
    old, new = "My office is in River.", "Correction: My office is in Lake, not River."
    await projected(ledger, "old", old, extracted(old, "River"), "2026-03-01T09:00:00+00:00")
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from test_source_claim_projection import Model
    ledger.record_source("new", contact_id="owner", session_id="s-1", occurred_at="2026-03-02T09:00:00+00:00",
                         messages=[{"role": "user", "content": new}])
    assert await SourceClaimProjection(ledger).process_one(Model({new: extracted(
        new, "Lake", operation="correct", match_prior=True)}))
    question = "Where is my office, River or Lake?"
    kept = await recalled(ledger, question, window=NOVEMBER)
    # The correction is kept, labelled: nothing dates it, so the window cannot place it.
    assert "Lake" in kept.content and '"validity_status": "unknown"' in kept.content.replace("'", '"')
    # The corrected message never comes back as a quotation of its own.
    assert old not in kept.content


# --- An explicit search's own window and reference time -----------------------------------------------------------

@pytest.fixture
def search_app(source_app, tmp_path, monkeypatch):  # noqa: F811 (fixture)
    from colonymind import vector
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    return source_app, ledger_at(tmp_path)


async def searched(app, query, **fields):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        return await client.post("/v1/host/memory/search", json={
            "identity": {"host_id": "fixture"}, "person_id": "owner", "session_id": "s-1", "query": query, **fields})


def iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


@pytest.mark.parametrize("window", [
    {"start": "2026-12-08T00:00:00+00:00", "end": "2026-12-01T00:00:00+00:00"},   # inverted
    {"start": "2026-12-01T00:00:00+00:00", "end": "2026-12-01T00:00:00+00:00"},   # empty
    {"start": "2026-12-01T00:00:00", "end": "2026-12-08T00:00:00"},               # no time zone
    {"start": "2026-12-01T00:00:00+00:00"},
])
async def test_a_window_that_is_not_a_span_is_refused(search_app, window):
    app, _ = search_app
    response = await searched(app, "What did we talk about?", window=window)
    assert response.status_code == 422, response.text


async def test_a_future_as_of_is_clamped_to_now(search_app):
    app, ledger = search_app
    now = datetime.now(UTC)
    record(ledger, "yesterday", "The kiln reached cone six.", iso(now - timedelta(days=1)))
    record(ledger, "earlier", "The ferry was late again.", iso(now - timedelta(days=11)))
    future = await searched(app, "What did I say yesterday?", as_of=iso(now + timedelta(days=30)))
    assert future.status_code == 200, future.text
    assert "cone six" in future.json()["content"] and "ferry" not in future.json()["content"]
    past = await searched(app, "What did I say yesterday?", as_of=iso(now - timedelta(days=10)))
    assert "ferry" in past.json()["content"] and "cone six" not in past.json()["content"]
    naive = await searched(app, "What did I say yesterday?", as_of="2026-12-01T00:00:00")
    assert naive.status_code == 422


async def test_a_supplied_window_adds_its_sources_and_never_removes_a_keyword_hit(search_app):
    app, ledger = search_app
    now = datetime.now(UTC)
    record(ledger, "gate", "The camera recorded a fox at the gate.", iso(now - timedelta(days=60)))
    record(ledger, "kiln", "The kiln reached cone six.", iso(now - timedelta(days=20)))
    window = {"start": iso(now - timedelta(days=21)), "end": iso(now - timedelta(days=19))}
    supplied = await searched(app, "What did the camera record at the gate?", window=window, trace=True)
    assert supplied.status_code == 200, supplied.text
    body = supplied.json()
    assert "fox at the gate" in body["content"] and "outside_requested_window" in body["content"]
    assert "cone six" in body["content"]
    kiln, = [row for row in body["trace"]["rows"] if row["turn_id"] == "kiln"]
    assert kiln["lanes"] == {"window": 1}                           # the trace shows the window lane's rank
    # The same question dated in its words keeps today's rule: an event span drops what falls outside it.
    parsed = await searched(app, "What did the camera record at the gate 20 days ago?")
    assert "fox at the gate" not in parsed.json()["content"] and "cone six" in parsed.json()["content"]
