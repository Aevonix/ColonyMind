"""A vague event date is kept as the whole period it names, and both claim readers accept it.

"in 2015" is the year and "last summer" the latest summer that had ended when the message was sent, in the
message's own time zone. A season resolves only where the time zone's location fixes the hemisphere; the rest
stays unresolved rather than guessed.
"""
from contextlib import closing
from datetime import datetime, timezone
import json

import pytest

from colonymind.beliefs.source_claims import validated_claims
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.beliefs.source_time import MemoryTimeQuery, interpret_time_query, source_event_time
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_projection import Model, claim
from test_source_episode_memory import episode

STAMP = "2026-10-05T12:00:00+00:00"


def period(expression, *, observed_at=STAMP, zone="UTC"):
    return source_event_time(expression, observed_at=observed_at, timezone_name=zone)


@pytest.mark.parametrize("expression,zone,start,end", [
    ("in 2015", "UTC", "2015-01-01T00:00:00+00:00", "2016-01-01T00:00:00+00:00"),
    ("2015", "Europe/Berlin", "2014-12-31T23:00:00+00:00", "2015-12-31T23:00:00+00:00"),
    ("during 2015", "America/New_York", "2015-01-01T05:00:00+00:00", "2016-01-01T05:00:00+00:00"),
])
def test_a_year_is_the_whole_year_in_the_source_timezone(expression, zone, start, end):
    assert period(expression, zone=zone) == {
        "expression": expression, "status": "resolved", "precision": "period", "period": "year",
        "start": start, "end_exclusive": end}
    # A stated year needs no clock: it resolves the same for a message whose time is unknown.
    assert period(expression, observed_at=None, zone=zone)["start"] == start


def test_a_month_with_its_year_is_the_whole_month_across_a_clock_change():
    result = period("in March 2015", zone="Europe/Berlin")
    assert result["precision"] == "period" and result["period"] == "month"
    # Summer time starts on 29 March 2015 in Berlin: the month ends at local midnight, 22:00 UTC.
    assert (result["start"], result["end_exclusive"]) == ("2015-02-28T23:00:00+00:00", "2015-03-31T22:00:00+00:00")
    assert period("March of 2015", zone="UTC")["start"] == "2015-03-01T00:00:00+00:00"


@pytest.mark.parametrize("observed_at,zone,start,end,hemisphere", [
    # After the summer: this year's.
    (STAMP, "Europe/Berlin", "2026-05-31T22:00:00+00:00", "2026-08-31T22:00:00+00:00", "northern"),
    # During the summer: the one before, which had ended.
    ("2026-07-15T12:00:00+00:00", "Europe/Berlin", "2025-05-31T22:00:00+00:00", "2025-08-31T22:00:00+00:00",
     "northern"),
    # The southern summer spans the new year: December to February, in local summer time.
    ("2026-07-15T00:00:00+00:00", "Australia/Sydney", "2025-11-30T13:00:00+00:00", "2026-02-28T13:00:00+00:00",
     "southern"),
])
def test_last_summer_is_the_latest_summer_that_had_ended_when_the_message_was_sent(
        observed_at, zone, start, end, hemisphere):
    result = period("last summer", observed_at=observed_at, zone=zone)
    assert result["status"] == "resolved" and result["precision"] == "period" and result["period"] == "season"
    assert (result["start"], result["end_exclusive"]) == (start, end)
    assert result["basis"]["hemisphere"] == hemisphere
    assert result["basis"]["observed_at"] == observed_at and result["basis"]["timezone_name"] == zone


@pytest.mark.parametrize("expression,observed_at,zone", [
    ("last summer", STAMP, "UTC"),                   # no location, no hemisphere
    ("last summer", STAMP, "Asia/Singapore"),        # the tropics have no summer to name
    ("last summer", None, "Europe/Berlin"),          # no message time to count back from
    ("last year", None, "Europe/Berlin"),
    ("winter 2015", STAMP, "Europe/Berlin"),         # a northern winter spans two years
    ("this summer", STAMP, "Europe/Berlin"),         # ahead or under way: not one period
    ("earlier this week", STAMP, "Europe/Berlin"),
    ("in 20155", STAMP, "UTC"),
])
def test_a_period_that_the_source_does_not_fix_stays_unresolved(expression, observed_at, zone):
    assert period(expression, observed_at=observed_at, zone=zone) == {"expression": expression,
                                                                       "status": "unresolved"}


def test_a_season_with_its_year_resolves_where_it_stays_in_that_year():
    summer = period("the summer of 2015", zone="Europe/Berlin")
    assert (summer["start"], summer["end_exclusive"]) == ("2015-05-31T22:00:00+00:00", "2015-08-31T22:00:00+00:00")
    winter = period("in the winter of 2015", zone="Australia/Sydney")
    assert (winter["start"], winter["end_exclusive"]) == ("2015-05-31T14:00:00+00:00", "2015-08-31T14:00:00+00:00")


def test_last_year_counts_back_from_the_local_date_of_the_message():
    # 21:00 on 31 December 2025 in New York is already 2026 in UTC.
    result = period("last year", observed_at="2026-01-01T02:00:00+00:00", zone="America/New_York")
    assert (result["start"], result["end_exclusive"]) == ("2024-01-01T05:00:00+00:00", "2025-01-01T05:00:00+00:00")
    assert result["basis"]["rule"] == "calendar_period_before_source_occurrence"
    month = period("last month", observed_at="2026-03-01T03:00:00+00:00", zone="America/New_York")
    assert (month["start"], month["end_exclusive"]) == ("2026-01-01T05:00:00+00:00", "2026-02-01T05:00:00+00:00")


def test_an_extracted_year_is_stored_as_a_span_with_no_point_in_time():
    text = "I saw the northern lights in 2015."
    row, = validated_claims(json.dumps([claim(text, "the northern lights", predicate="sighting",
                                              event_at_text="in 2015")]),
                            message=text, prior=[], observed_at=STAMP, timezone_name="Europe/Berlin")
    assert row["event_at"] is None
    assert row["event_time"]["precision"] == "period"
    assert row["event_time"]["start"] == "2014-12-31T23:00:00+00:00"


def test_an_extracted_last_summer_is_a_span_relative_to_the_message():
    text = "We hiked the ridge trail last summer."
    row, = validated_claims(json.dumps([claim(text, "the ridge trail", subject="We", predicate="hike",
                                              event_at_text="last summer")]),
                            message=text, prior=[], observed_at=STAMP, timezone_name="Europe/Berlin")
    assert row["event_at"] is None
    assert (row["event_time"]["start"], row["event_time"]["end_exclusive"]) == (
        "2026-05-31T22:00:00+00:00", "2026-08-31T22:00:00+00:00")


def rows_for(projection, query, *, now="2026-10-05T12:00:00+00:00"):
    hits = projection.ledger.search_sources(query, contact_id="contact-a", session_id="later", limit=10)
    _, rows = projection.prepare_context([], hits, contact_id="contact-a", session_id="later",
        time_query=interpret_time_query(query, now=datetime.fromisoformat(now), timezone_name="UTC"))
    return [row for row in rows if row.get("atomic_evidence")]


@pytest.mark.asyncio
async def test_a_year_span_answers_a_date_inside_that_year_through_both_readers(tmp_path):
    text = "I saw the northern lights in 2015."
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    ledger.record_source("lights", contact_id="contact-a", session_id="s", occurred_at=STAMP,
                         messages=[{"role": "user", "content": text}])
    assert await projection.process_one(Model({text: claim(text, "the northern lights", predicate="sighting",
                                                           event_at_text="in 2015")}))
    inside = interpret_time_query("When were the northern lights seen on 12 March 2015?",
                                  now=datetime(2026, 10, 5, tzinfo=timezone.utc))
    outside = interpret_time_query("When were the northern lights seen on 12 March 2016?",
                                   now=datetime(2026, 10, 5, tzinfo=timezone.utc))
    assert inside.mode == outside.mode == "observed_range"
    with closing(ledger._connect()) as conn:
        row, = projection._rows(conn, "contact-a", "later")
        key = (row["subject_key"], row["predicate"])
        assert inside.accepts_claim(row) and not outside.accepts_claim(row)
        assert [r["id"] for r in projection._rows(conn, "contact-a", "later", key=key, time_query=inside)] == [row["id"]]
        assert projection._rows(conn, "contact-a", "later", key=key, time_query=outside) == []
    # Recall offers it for a day inside the year, but a year cannot certify the day.
    card, = rows_for(projection, "When were the northern lights seen on 12 March 2015?")
    assert json.loads(card["content"])["assertions"][0]["event_time"]["precision"] == "period"
    assert card["validity_status"] == "query_time_unresolved"
    assert rows_for(projection, "When were the northern lights seen on 12 March 2016?") == []


def test_a_period_that_contains_the_whole_window_is_accepted_and_one_outside_is_not():
    year = {"event_time": source_event_time("2015", observed_at=None, timezone_name="UTC")}
    assert MemoryTimeQuery("observed_range", "2015-06-01T00:00:00+00:00", "2015-06-02T00:00:00+00:00").accepts_claim(year)
    assert not MemoryTimeQuery("observed_range", "2016-01-01T00:00:00+00:00", "2016-01-02T00:00:00+00:00").accepts_claim(year)


@pytest.mark.asyncio
async def test_an_episode_dated_to_another_year_is_not_offered_as_an_undated_one(tmp_path):
    text = "We watched a comet from the harbour pier in 2015; it lit the whole bay for an hour."
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    ledger.record_source("comet", contact_id="contact-a", session_id="s", occurred_at=STAMP,
                         messages=[{"role": "user", "content": text}])
    assert await projection.process_one(Model({text: episode(text) | {"event_at_text": "in 2015"}}))
    assert rows_for(projection, "What was seen from the harbour pier on 12 March 2016?") == []
    card, = rows_for(projection, "What was seen from the harbour pier on 12 March 2015?")
    assert card["validity_status"] == "query_time_unresolved"


# -- one period is not another time that starts with it ---------------------------------------------------------

DAY, YEAR, MONTH = ("I saw the northern lights on 2015-01-01.", "I saw the northern lights in 2015.",
                    "I saw the northern lights in January 2015.")


@pytest.mark.asyncio
async def test_a_day_a_month_and_a_year_that_start_together_are_three_priors_in_either_lane(tmp_path):
    from colonymind.beliefs.source_projection import interleave_lanes, one_witness_per_value, witness_key
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    dates = {DAY: "2015-01-01", YEAR: "in 2015", MONTH: "in January 2015"}
    model = Model({text: claim(text, "the northern lights", predicate="sighting", event_at_text=when)
                   for text, when in dates.items()})
    for index, text in enumerate(dates):
        ledger.record_source(f"lights-{index}", contact_id="contact-a", session_id=f"s{index}",
                             occurred_at=f"2026-10-0{index + 1}T12:00:00+00:00",
                             messages=[{"role": "user", "content": text}])
        assert await projection.process_one(model)
    with closing(ledger._connect()) as conn:
        rows = projection._rows(conn, "contact-a", "later")
    assert len(rows) == 3
    starts = {row["event_time"]["start"] for row in rows}
    assert starts == {"2015-01-01T00:00:00+00:00"}               # the same first instant, three different times
    assert len({witness_key(row) for row in rows}) == 3
    # Within a lane: each claim stays a prior the extractor can name.
    assert len(one_witness_per_value(rows)) == 3
    source = {"turn_id": "new", "contact_id": "contact-a", "session_id": "later"}
    offered = projection.prior(source, {"role": "user", "content": "The northern lights were a mistake, never mind."})
    assert {row["id"] for row in offered} == {row["id"] for row in rows}
    # Between lanes: a day in one and the year in the other are both offered.
    day, year = (next(row for row in rows if row["evidence"] == text) for text in (DAY, YEAR))
    assert [row["id"] for row in interleave_lanes([day], [year], limit=16)] == [day["id"], year["id"]]
    # The same period said twice is still one value.
    again = dict(year, id="claim:again", event_time=dict(year["event_time"], expression="during 2015"))
    assert witness_key(again) == witness_key(year)
    assert [row["id"] for row in interleave_lanes([year], [again], limit=16)] == [year["id"]]


@pytest.mark.asyncio
async def test_the_atlas_shows_a_period_as_the_span_it_is_never_as_its_first_day(tmp_path):
    from colonymind.atlas import build_atlas
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    projection = SourceClaimProjection(ledger)
    comet = "I saw a comet on 2015-03-12."
    for index, (text, when) in enumerate(((YEAR, "in 2015"), (comet, "2015-03-12"),
                                          ("I saw an eclipse in March 2016.", "in March 2016"))):
        ledger.record_source(f"sky-{index}", contact_id="contact-a", session_id="s", occurred_at=STAMP,
                             messages=[{"role": "user", "content": text}])
        value = text.removeprefix("I saw ").split(" in ")[0].split(" on ")[0]
        assert await projection.process_one(Model({text: claim(text, value, predicate="sighting",
                                                               event_at_text=when)}))
    out = tmp_path / "atlas"
    build_atlas(ledger_path=tmp_path / "turn-idempotency.db", out_dir=out, owner_id="contact-a",
                owner_name="Owner", now=datetime(2026, 10, 6, tzinfo=timezone.utc))
    pages = "\n".join(path.read_text() for path in sorted(out.rglob("*.html")))
    assert "happened 2015-01-01" not in pages and "happened 2016-03-01" not in pages
    assert "happened sometime between 2015-01-01 and 2015-12-31" in pages
    assert "happened sometime between 2016-03-01 and 2016-03-31" in pages
    assert "happened 2015-03-12" in pages                      # a stated day is still that day
