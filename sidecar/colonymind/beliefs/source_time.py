"""Explicit valid-time and observation-time windows for memory recall."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
from zoneinfo import TZPATH, ZoneInfo

UTC = timezone.utc
_MONTHS = {name.casefold(): i for i, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1)}
_DATE = r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2}))?"
_MONTH = r"(?:" + "|".join(_MONTHS) + r")"
_MONTH_DATE = (r"(?:" + _MONTH + r"\s+\d{1,2},?\s+\d{4}|"
               r"\d{1,2}\s+" + _MONTH + r"\s+\d{4})")
_TIME_ON_DATE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    r"(?:\s+(?P<utc>UTC)|\s*(?P<offset>[+-]\d{2}:\d{2}))?"
    r"\s+on\s+(?P<date>" + _MONTH_DATE + r")", re.I)
# Recognize a clock-shaped operand even when its meridiem/zone is unsupported.
# Parsing remains limited to _TIME_ON_DATE; never reduce a rejected clock to
# its embedded calendar date or discard a quoted operand as ordinary evidence.
_CLOCK_DATE_OPERAND = (r"\b\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s+[a-z][a-z0-9_./+:-]{0,63}|\s*[+-][a-z0-9_./+:-]{1,63}){0,2}"
    r"\s+on\s+" + _MONTH_DATE)
_EXPLICIT_DATE = _CLOCK_DATE_OPERAND + "|" + _DATE + "|" + _MONTH_DATE
_EVENT = re.compile(r"\b(footage|camera|observed|spotted|seen|saw|happened|recorded|arrived|visited)\b", re.I)


def utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(UTC) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def _english_source_date(value: str, zone) -> datetime | None:
    """Literal full-month dates, optionally preceded by a 24-hour clock.

    UTC or a numeric HH:MM offset overrides the profile zone. An unqualified
    clock uses it, but cannot resolve a repeated or nonexistent local time.
    """
    clock = _TIME_ON_DATE.fullmatch(value)
    date_text = clock['date'] if clock else value
    match = re.fullmatch(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_text)
    if match:
        month, day, year = match[1], match[2], match[3]
    else:
        match = re.fullmatch(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", date_text)
        if not match:
            return None
        day, month, year = match[1], match[2], match[3]
    if month not in _MONTHS:
        return None
    naive = datetime(int(year), _MONTHS[month], int(day),
                     int(clock['hour']) if clock else 0,
                     int(clock['minute']) if clock else 0,
                     int(clock['second'] or 0) if clock else 0)
    selected_zone = UTC if clock and clock['utc'] else zone
    if clock and clock['offset']:
        offset = clock['offset']
        hours, minutes = map(int, offset[1:].split(':'))
        if hours > 23 or minutes > 59:
            return None
        delta = timedelta(hours=hours, minutes=minutes)
        selected_zone = timezone(-delta if offset[0] == '-' else delta)
    aware = naive.replace(tzinfo=selected_zone)
    if clock and (aware.utcoffset() != aware.replace(fold=1).utcoffset()
                  or aware.astimezone(UTC).astimezone(selected_zone).replace(tzinfo=None) != naive):
        return None
    return aware.astimezone(UTC)


def parse_source_date(expression: str, *, observed_at: str | None, timezone_name="UTC") -> str | None:
    """Resolve only explicit supported date text, anchored to source occurrence."""
    zone = ZoneInfo(timezone_name)
    value = expression.strip().casefold()
    try:
        if value in {"now", "today", "yesterday", "tomorrow"}:
            observed = utc_timestamp(observed_at)
            if observed is None:
                return None
            if value == "now":
                return observed.isoformat()
            local = observed.astimezone(zone).replace(hour=0, minute=0, second=0, microsecond=0)
            local += timedelta(days={"yesterday": -1, "today": 0, "tomorrow": 1}[value])
            return local.astimezone(UTC).isoformat()
        if re.fullmatch(_DATE, expression, re.I):
            date = datetime.fromisoformat(expression.replace("Z", "+00:00"))
            if date.tzinfo is None:
                date = date.replace(tzinfo=zone)
            return date.astimezone(UTC).isoformat()
        parsed = _english_source_date(value, zone)
        if parsed is not None:
            return parsed.isoformat()
    except ValueError:
        pass
    return None


#: Meteorological seasons: three whole months each, starting in these months north of the tropics. South of
#: them each season starts six months later. Near the equator no season is named by the calendar.
_SEASON_STARTS = {"spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12}
_TROPIC_LATITUDE = 23.44
_SEASON = r"(?P<season>spring|summer|autumn|fall|winter)"
_IN = r"(?:(?:in|during)\s+)?"
_PERIOD_FORMS = (
    ("year", re.compile(_IN + r"(?P<year>\d{4})")),
    ("month", re.compile(_IN + r"(?P<month>" + _MONTH + r")(?:\s+of)?,?\s+(?P<year>\d{4})")),
    ("season", re.compile(_IN + r"(?:the\s+)?" + _SEASON + r"(?:\s+of)?,?\s+(?P<year>\d{4})")),
    ("relative", re.compile(_IN + r"(?P<which>last|this)\s+(?P<unit>year|month)")),
    ("last_season", re.compile(_IN + r"last\s+" + _SEASON)),
)


@lru_cache(maxsize=1)
def _zone_latitudes() -> dict[str, float]:
    """Each time zone's location latitude, from the time zone database's own location tables.

    A zone the tables do not list (``UTC``, ``Etc/*``, an alias) has none, and neither has an installation
    without the tables: a season then stays unresolved.
    """
    candidates = [Path(base) / name for base in TZPATH for name in ("zone1970.tab", "zone.tab")]
    try:
        from importlib.resources import files
        candidates += [files("tzdata").joinpath("zoneinfo", name) for name in ("zone1970.tab", "zone.tab")]
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    latitudes: dict[str, float] = {}
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        for line in text.splitlines():
            fields = line.split("\t")
            if line.startswith("#") or len(fields) < 3:
                continue
            match = re.match(r"([+-])(\d{2})(\d{2})(\d{2})?[+-]", fields[1])
            if match:
                latitude = int(match[2]) + int(match[3]) / 60 + int(match[4] or 0) / 3600
                latitudes.setdefault(fields[2].strip(), -latitude if match[1] == "-" else latitude)
    return latitudes


def season_hemisphere(timezone_name: str) -> str | None:
    """``northern`` or ``southern`` where the time zone's location lies outside the tropics; otherwise None."""
    latitude = _zone_latitudes().get(timezone_name)
    if latitude is None or abs(latitude) < _TROPIC_LATITUDE:
        return None
    return "northern" if latitude > 0 else "southern"


def _local_month_start(zone, year: int, month: int) -> datetime:
    """Local midnight on the first of a month, in UTC. ``month`` may run past December."""
    year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
    return datetime(year, month, 1, tzinfo=zone).astimezone(UTC)


def source_event_period(expression: str, *, observed_at: str | None, timezone_name="UTC") -> dict | None:
    """A vague event date as the whole period it names (``precision: period``), or None.

    A year ("in 2015"), a month with its year ("March 2015"), a season with its year ("the summer of 2015")
    and the periods counted back from the message ("last year", "last month", "this year", "this month",
    "last summer"). Periods are whole local months in the message's time zone, as UTC instants with the end
    exclusive. "last <season>" is the latest such season that had ended when the message was sent. A season
    resolves only where the time zone fixes the hemisphere, and one with a year only where it lies inside
    that year. Anything else is None: the caller keeps it unresolved.
    """
    value = " ".join(expression.casefold().split())
    zone = ZoneInfo(timezone_name)
    observed = utc_timestamp(observed_at)
    local = observed.astimezone(zone) if observed else None
    for kind, pattern in _PERIOD_FORMS:
        match = pattern.fullmatch(value)
        if match is None:
            continue
        year = int(match["year"]) if "year" in match.groupdict() and match["year"] else None
        if year is not None and not 1 <= year < 9999:
            return None
        basis = None
        hemisphere = season_hemisphere(timezone_name) if "season" in match.groupdict() else None
        if kind == "year":
            period, first, months = "year", (year, 1), 12
        elif kind == "month":
            period, first, months = "month", (year, _MONTHS[match["month"]]), 1
        elif kind == "relative":
            if local is None:
                return None
            period = match["unit"]
            back = 1 if match["which"] == "last" else 0
            first = (local.year - back, 1) if period == "year" else (local.year, local.month - back)
            months = 12 if period == "year" else 1
            basis = {"rule": ("calendar_period_before_source_occurrence" if back
                              else "calendar_period_of_source_occurrence")}
        else:
            if hemisphere is None:
                return None
            period, months = "season", 3
            start_month = (_SEASON_STARTS[match["season"]] + (6 if hemisphere == "southern" else 0) - 1) % 12 + 1
            if kind == "season":
                if start_month + months - 1 > 12:
                    return None  # It spans two years: the year alone does not say which.
                first = (year, start_month)
                basis = {"hemisphere": hemisphere, "seasons": "meteorological"}
            else:
                if local is None:
                    return None
                ended = [(y, start_month) for y in (local.year, local.year - 1, local.year - 2)
                         if _local_month_start(zone, y, start_month + months) <= observed]
                if not ended:
                    return None
                first = ended[0]
                basis = {"rule": "latest_completed_season_before_source_occurrence", "hemisphere": hemisphere,
                         "seasons": "meteorological"}
        if basis is not None and basis.get("rule"):
            basis.update(observed_at=observed_at, timezone_name=timezone_name)
        start = _local_month_start(zone, *first)
        end = _local_month_start(zone, first[0], first[1] + months)
        return {"expression": expression, "status": "resolved", "precision": "period", "period": period,
                "start": start.isoformat(), "end_exclusive": end.isoformat(), **({"basis": basis} if basis else {})}
    return None


def event_interval(event) -> tuple[datetime, datetime] | None:
    """A resolved event's interval (start, end exclusive) in UTC: a calendar day, a period, or any other
    interval precision. None for an instant, a bound or an unknown time."""
    if not isinstance(event, dict) or event.get("status") != "resolved":
        return None
    start, end = utc_timestamp(event.get("start")), utc_timestamp(event.get("end_exclusive"))
    return (start, end) if start and end else None


def source_event_time(expression: str | None, *, observed_at: str | None, timezone_name="UTC") -> dict:
    """Retain source-grounded event precision without inventing a point in time.

    The caller verifies the expression against the quotation. An unresolved
    optional event date is different from an unresolved state-validity condition.
    The latter must still prevent certifying when a state holds. A vague date
    is the whole period it names (``source_event_period``), never its first day.
    """
    if expression is None:
        return {"status": "unknown"}
    result = {"expression": expression}
    parsed = parse_source_date(expression, observed_at=observed_at, timezone_name=timezone_name)
    if parsed:
        if (expression.casefold().strip() == "now"
                or (re.fullmatch(_DATE, expression, re.I) and "T" in expression.upper())
                or _TIME_ON_DATE.fullmatch(expression.strip())):
            return {**result, "status": "resolved", "precision": "instant", "at": parsed}
        begin = utc_timestamp(parsed).astimezone(ZoneInfo(timezone_name))
        return {**result, "status": "resolved", "precision": "calendar_day", "start": parsed,
                "end_exclusive": (begin + timedelta(days=1)).astimezone(UTC).isoformat()}
    try:
        span = source_event_period(expression, observed_at=observed_at, timezone_name=timezone_name)
    except (ValueError, OverflowError):
        span = None
    if span is not None:
        return span
    # This relation refers to the message's occurrence, never its later import.
    if re.fullmatch(r"before (?:sending )?(?:this|the current) message", expression.strip(), re.I):
        observed = utc_timestamp(observed_at)
        if observed:
            return {**result, "status": "bounded", "before": observed.isoformat()}
    return {**result, "status": "unresolved"}


_WEEKDAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
_DEADLINE_DAY = r'(?P<day>' + _MONTH_DATE + r'|\d{4}-\d{2}-\d{2}|today|tomorrow|' + '|'.join(_WEEKDAYS) + ')'
_DEADLINE_CLOCK = r'(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)?'
_DEADLINE_PART = r'(?:\s+(?P<part>morning|afternoon|evening))?'
_DEADLINE_FORMS = tuple(re.compile(pattern, re.I) for pattern in (
    _DEADLINE_CLOCK + r'\s+(?:on\s+)?' + _DEADLINE_DAY + _DEADLINE_PART,
    _DEADLINE_DAY + _DEADLINE_PART + r'\s+(?:at\s+)?' + _DEADLINE_CLOCK))
_UNRESOLVED_RELATIVE_DEADLINE = re.compile(
    r'\b(?:was|were|had|last|previous|ago|yesterday|earlier|before|'
    r'happened|occurred|departed|arrived|left|attended|missed|finished|ended|'
    r'not|cancelled|canceled|if|unless|maybe|might|or)\b', re.I)
_DEADLINE_ASSERTION = (r'\b(?:is|are|starts?|begins?|departs?|leaves?|due|scheduled|will be)'
                       r'\s+(?:(?:at|on|for)\s+)?')


def source_deadline_time(expression: str, *, observed_at: str | None, timezone_name="UTC",
                         evidence: str | None = None) -> dict:
    """Interpret a selected deadline operand without changing event memories.

    An ordinary clock may accompany an explicit date, today or tomorrow. In a
    deadline read, a bare weekday means its next distinct occurrence after the
    report's local day. Same-day weekdays are ambiguous between this and next
    week and stay unresolved. Neither ingestion nor the current clock is used.
    """
    result = source_event_time(expression, observed_at=observed_at, timezone_name=timezone_name)
    if result['status'] != 'unresolved':
        return result
    match = next((m for pattern in _DEADLINE_FORMS if (m := pattern.fullmatch(expression.strip()))), None)
    if match is None or not (match['meridiem'] or match['minute']):
        return result
    hour, minute = int(match['hour']), int(match['minute'] or 0)
    meridiem = (match['meridiem'] or '').lower()
    if (minute > 59 or (meridiem and not 1 <= hour <= 12) or (not meridiem and hour > 23)):
        return result
    if meridiem:
        hour = hour % 12 + (12 if meridiem == 'pm' else 0)
    part = (match['part'] or '').lower()
    if part and ((part == 'morning') != (hour < 12)):
        return result
    zone, day = ZoneInfo(timezone_name), match['day'].lower()
    basis = None
    if day in {*_WEEKDAYS, 'today', 'tomorrow'}:
        # A trimmed event operand cannot hide a past report, cancellation or
        # condition in its full selected quotation. Ambiguous prose remains a
        # readable claim but cannot establish this forward interpretation.
        if (not evidence or expression not in evidence
                or _UNRESOLVED_RELATIVE_DEADLINE.search(evidence)
                or not re.search(_DEADLINE_ASSERTION + re.escape(expression), evidence, re.I)):
            return result
    if day in _WEEKDAYS:
        reported = utc_timestamp(observed_at)
        if reported is None:
            return result
        local = reported.astimezone(zone)
        distance = (_WEEKDAYS.index(day) - local.weekday()) % 7
        if not distance:
            return result
        date = local.date() + timedelta(days=distance)
        basis = {'rule': 'next_distinct_weekday_from_source_occurrence',
                 'observed_at': observed_at, 'timezone_name': timezone_name}
    else:
        parsed = parse_source_date(day, observed_at=observed_at, timezone_name=timezone_name)
        if parsed is None:
            return result
        date = utc_timestamp(parsed).astimezone(zone).date()
        if day in {'today', 'tomorrow'}:
            basis = {'rule': 'relative_day_from_source_occurrence',
                     'observed_at': observed_at, 'timezone_name': timezone_name}
    naive = datetime(date.year, date.month, date.day, hour, minute)
    aware = naive.replace(tzinfo=zone)
    if (aware.utcoffset() != aware.replace(fold=1).utcoffset()
            or aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != naive):
        return result
    return {'expression': expression, 'status': 'resolved', 'precision': 'instant',
            'at': aware.astimezone(UTC).isoformat(), **({'basis': basis} if basis else {})}


#: The start of a span open to the past ("before 20 December 2026"): the earliest instant, in the form every store
#: compares with, so a claim or a source of any date falls after it.
EARLIEST = "0001-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class MemoryTimeQuery:
    """The time a recall asks about. ``mode``: ``current``, ``observed_range`` (when something happened or was
    recorded), ``valid_range`` (when a fact held) or ``unresolved_time`` (a time was asked about and not
    understood: no claim is admitted). A range is ``start`` to ``end``, the end exclusive. ``supplied``: the range
    came from the caller's structured window, not the words; it sets the claim view and adds the date-window lane,
    and never removes a quotation (``filter_unstructured``)."""
    mode: str = "current"
    start: str | None = None
    end: str | None = None
    expression: str | None = None
    supplied: bool = False

    def window(self) -> tuple[datetime, datetime] | None:
        """The span the date-window lane reads (``memory.time_lane``): a resolved range, else None."""
        if self.mode not in ("observed_range", "valid_range"):
            return None
        start, end = utc_timestamp(self.start), utc_timestamp(self.end)
        return (start, end) if start is not None and end is not None and start < end else None

    def accepts_observation(self, occurred_at: str | None) -> bool:
        if self.mode != "observed_range":
            return True
        stamp = utc_timestamp(occurred_at)
        if stamp is None:
            return False
        return (not self.start or stamp >= utc_timestamp(self.start)) and (not self.end or stamp < utc_timestamp(self.end))

    def accepts_claim(self, claim: dict) -> bool:
        if claim.get("retracted_by"):
            return False
        if self.mode == "unresolved_time":
            return False
        if self.mode == "observed_range":
            # Any resolved interval (a calendar day, a period) overlaps the window; a point uses event_at.
            interval = event_interval(claim.get('event_time'))
            if interval is not None:
                start, end = interval
                return ((not self.end or start < utc_timestamp(self.end))
                        and (not self.start or end > utc_timestamp(self.start)))
            return self.accepts_observation(claim.get("event_at"))
        start, end = utc_timestamp(claim.get("valid_from")), utc_timestamp(claim.get("valid_to"))
        query_start, query_end = utc_timestamp(self.start), utc_timestamp(self.end)
        if self.mode == "valid_range":
            # A historical assertion with no known validity start cannot be
            # certified as true on an arbitrary earlier date.
            if start is None:
                return False
            return (query_end is None or start < query_end) and (end is None or query_start < end)
        if query_start:
            return (start is None or start <= query_start) and (end is None or query_start < end)
        return not claim.get("superseded_by")


# Relative dates a question can name, resolved against the turn's own time (``_relative_span``). Counts in words go
# to twelve; "a week ago" is one.
_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
# A number is read only whole: never the "5" of "1.5" or the "000" of "1,000" (those are ``_UNSUPPORTED``).
_COUNT = r"(?:(?<![\d.,])\d{1,3}|" + "|".join(_COUNTS) + r")"
_SEASON_WORD = r"(?:spring|summer|autumn|fall|winter)"
_RELATIVE = re.compile(
    r"\b(?:" + _COUNT + r"\s+(?:day|week|month|year)s?\s+ago"
    r"|(?:the\s+(?:last|past)|last|this)\s+(?:week|month|year)"
    r"|(?:(?:last|this)\s+|(?:over\s+)?the\s+)weekend"
    r"|last\s+" + _SEASON_WORD
    + r"|(?:(?:in|during)\s+)?(?:the\s+)?" + _SEASON_WORD + r"(?:\s+of)?,?\s+\d{4}"
    r"|(?:(?:in|during)\s+)?" + _MONTH + r"(?:\s+of)?,?\s+\d{4}"
    r"|(?:in|during)\s+" + _MONTH + r"(?!\s*,?\s*\d)"
    r"|(?:in|during)\s+\d{4}"
    r")\b", re.I)
_EXPLICIT = re.compile(_EXPLICIT_DATE + r"|\b(?:today|yesterday|tomorrow)\b", re.I)
# A question year outside these is more likely a count ("in 1500 steps") than a year (my choice).
_QUESTION_YEARS = range(1900, 2200)
# Ranges and relative dates the parser does not resolve: a question naming one admits no claim.
# "N units ago" with a count the parser does not read (a decimal, a separator, four digits or more) is one.
_UNSUPPORTED = re.compile(r"\b(?:after|until|through|(?:next|previous)\s+(?:week|month|year))\b"
                          r"|(?<![\w.,])(?:\d+[.,][\d.,]*\d|\d{4,})\s+(?:day|week|month|year)s?\s+ago\b", re.I)


def _local_day(zone, day) -> datetime:
    """Local midnight at the start of ``day``, in UTC."""
    return datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(UTC)


def _relative_span(expression: str, *, now: datetime, timezone_name: str) -> tuple[datetime, datetime] | None:
    """The span a relative date names, from the turn's own time ``now`` in the contact's zone: local calendar days,
    weeks (Monday to Monday), months and years, end exclusive, as UTC instants. None when it names none the zone
    can place (a season where the zone fixes no hemisphere) or when it is out of range.

    - "last week", "this week": the calendar week before this one, or this one;
    - "the last week", "the past month": the trailing week, month or year up to ``now``;
    - "N days ago": that local day; "N weeks ago": the calendar week that held the day 7N days back; "N months
      ago", "N years ago": that calendar month or year;
    - "the weekend", "over the weekend": the latest weekend (Saturday and Sunday) that has begun; "last weekend": the
      one before it while a weekend is under way, else the same; "this weekend": this week's;
    - "in March": the latest March that has begun;
    - "last month", "last year", "this month", "this year", "last summer", "March 2015", "the summer of 2015",
      "in 2015": the period ``source_event_period`` reads, a season only where the zone fixes the hemisphere.
    """
    zone = ZoneInfo(timezone_name)
    value = " ".join(expression.casefold().split())
    local = now.astimezone(zone)
    today = local.date()
    try:
        if match := re.fullmatch(r"(" + _COUNT + r") (day|week|month|year)s? ago", value):
            count = _COUNTS.get(match[1]) or int(match[1])
            if match[2] == "day":
                first = today - timedelta(days=count)
                return _local_day(zone, first), _local_day(zone, first + timedelta(days=1))
            if match[2] == "week":
                day = today - timedelta(days=7 * count)
                monday = day - timedelta(days=day.weekday())
                return _local_day(zone, monday), _local_day(zone, monday + timedelta(days=7))
            first, months = (((local.year, local.month - count), 1) if match[2] == "month"
                             else ((local.year - count, 1), 12))
            return _local_month_start(zone, *first), _local_month_start(zone, first[0], first[1] + months)
        if match := re.fullmatch(r"the (?:last|past) (week|month|year)", value):
            if match[1] == "week":
                return now - timedelta(days=7), now
            back = 1 if match[1] == "month" else 12
            year, month = local.year + (local.month - 1 - back) // 12, (local.month - 1 - back) % 12 + 1
            day = min(local.day, (_local_month_start(zone, year, month + 1).astimezone(zone)
                                  - timedelta(days=1)).day)
            return local.replace(year=year, month=month, day=day).astimezone(UTC), now
        if match := re.fullmatch(r"(last|this) week", value):
            monday = today - timedelta(days=today.weekday() + (7 if match[1] == "last" else 0))
            return _local_day(zone, monday), _local_day(zone, monday + timedelta(days=7))
        if match := re.fullmatch(r"(?:(last|this) |(?:over )?the )weekend", value):
            if match[1] == "this":
                saturday = today - timedelta(days=today.weekday()) + timedelta(days=5)
            else:
                saturday = today - timedelta(days=(today.weekday() - 5) % 7)
                if match[1] == "last" and today.weekday() >= 5:
                    saturday -= timedelta(days=7)
            return _local_day(zone, saturday), _local_day(zone, saturday + timedelta(days=2))
        if match := re.fullmatch(r"(?:in|during) (" + _MONTH + r")", value):
            month = _MONTHS[match[1]]
            year = local.year if month <= local.month else local.year - 1
            return _local_month_start(zone, year, month), _local_month_start(zone, year, month + 1)
        year = re.search(r"\d{4}", value)
        if year and int(year[0]) not in _QUESTION_YEARS:
            return None
        span = source_event_period(value, observed_at=now.astimezone(UTC).isoformat(), timezone_name=timezone_name)
    except (ValueError, OverflowError):
        return None
    if span is None:
        return None
    return utc_timestamp(span["start"]), utc_timestamp(span["end_exclusive"])


def _temporal_request_text(text: str) -> str:
    """Exclude delimited evidence, retaining a quoted date used as an operand.

    This only selects text for temporal interpretation. Retrieval still uses
    the original query, including every quoted word and structured field. A
    leading arrival stamp is when the message came, never a date it asks about.
    """
    from colonymind.util.temporal import strip_arrival_stamps
    # Read as a date, an arrival stamp limited every stamped question to its own arrival day.
    text = strip_arrival_stamps(text)
    text = re.sub(r"(?s)```.*?```|~~~.*?~~~", " ", text)
    text = re.sub(r"(?m)^\s*>[^\n]*", " ", text)
    decoder = json.JSONDecoder()
    spans, through = [], 0
    for match in re.finditer(r"[\[{]", text):
        if match.start() < through:
            continue
        try:
            value, end = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            spans.append((match.start(), end))
            through = end
    for start, end in reversed(spans):
        text = text[:start] + " " + text[end:]

    def quoted(match):
        value = match[0][1:-1].strip()
        if (re.fullmatch(_EXPLICIT_DATE + "|today|yesterday|tomorrow", value, re.I)
                or _RELATIVE.fullmatch(value)
                or re.fullmatch(r"last\s+\d{1,3}\s+(?:hours?|days?)|"
                    r"(?:last|next|previous)\s+(?:week|month|year)|"
                    r"\d[\d.,]*\s+(?:days?|weeks?|months?|years?)\s+ago", value, re.I)):
            return value
        return " "

    return re.sub(r'''"(?:\\.|[^"\\])*"|“[^”]*”|(?<!\w)'(?:\\.|[^'\\])*'(?!\w)|`[^`\n]*`''', quoted, text)


def _operands(text: str) -> list[re.Match]:
    """The dates a question names, in order: explicit dates first, then relative ones that do not overlap them."""
    explicit = list(_EXPLICIT.finditer(text))
    relative = [match for match in _RELATIVE.finditer(text)
                if not any(match.start() < other.end() and other.start() < match.end() for other in explicit)]
    return sorted(explicit + relative, key=lambda match: match.start())


def _operand_span(match: re.Match, *, now: datetime, timezone_name: str) -> tuple[datetime, datetime] | None | str:
    """An operand's span; None when it is a relative date the zone cannot place (it is then not read);
    ``"unresolved"`` for an explicit date that does not resolve."""
    expression = match[0]
    if _EXPLICIT.fullmatch(expression) is None:
        return _relative_span(expression, now=now, timezone_name=timezone_name)
    start = parse_source_date(expression, observed_at=now.isoformat(), timezone_name=timezone_name)
    if not start:
        return "unresolved"
    begin = utc_timestamp(start)
    if _TIME_ON_DATE.fullmatch(expression) or (re.fullmatch(_DATE, expression, re.I) and "T" in expression.upper()):
        return begin, begin + timedelta(microseconds=1)
    return begin, (begin.astimezone(ZoneInfo(timezone_name)) + timedelta(days=1)).astimezone(UTC)


def interpret_time_query(text: str, *, now: datetime, timezone_name="UTC",
                         window: tuple[datetime, datetime] | None = None) -> MemoryTimeQuery:
    """Calendar days differ from trailing windows and from source ingestion.

    A date without a clock covers that local calendar day. If a fact changes
    during the day, both intersecting validity intervals remain visible. A
    relative date ("last week", "in March", "3 days ago") is the span it names,
    counted from ``now`` (the turn's own time, or a search's ``as_of``) in the
    contact's zone (``_relative_span``). "since X" runs from X to ``now``,
    "before X" from the earliest instant to X, "between X and Y" from X to the
    end of Y. Any other range, several dates, or a date that does not resolve
    is ``unresolved_time``. A caller's ``window`` (start, end exclusive) is
    the span, whatever the words say, and is marked ``supplied``.
    """
    zone = ZoneInfo(timezone_name)
    text = _temporal_request_text(text[:4096])
    mode = "observed_range" if _EVENT.search(text) else "valid_range"
    if window is not None:
        start, end = (stamp.astimezone(UTC).isoformat() for stamp in window)
        return MemoryTimeQuery(mode, start, end, supplied=True)
    duration = re.search(r"\blast\s+(\d{1,3})\s+(hours?|days?)\b", text, re.I)
    if duration and mode == "observed_range":
        delta = timedelta(hours=int(duration[1])) if duration[2].lower().startswith("hour") else timedelta(days=int(duration[1]))
        return MemoryTimeQuery(mode, (now - delta).astimezone(UTC).isoformat(), now.astimezone(UTC).isoformat(), duration[0])
    # Do not quietly turn unsupported ranges/relative dates into a current
    # answer or select the first date in a multi-date question.
    if _UNSUPPORTED.search(text):
        return MemoryTimeQuery("unresolved_time", expression=text)
    spans = []
    for match in _operands(text):
        span = _operand_span(match, now=now, timezone_name=timezone_name)
        if span == "unresolved":
            return MemoryTimeQuery("unresolved_time", expression=match[0])
        if span is not None:
            spans.append((match, span))
    between = re.search(r"\bbetween\b\s*", text, re.I)
    before = re.search(r"\bbefore\b\s*", text, re.I)
    if between or before:
        connector = between or before
        if (between and before) or len(spans) != (2 if between else 1) or spans[0][0].start() != connector.end():
            return MemoryTimeQuery("unresolved_time", expression=text)
        if between:
            (first, (start, _)), (second, (_, end)) = spans
            if not re.fullmatch(r"\s+and\s+", text[first.end():second.start()], re.I) or start >= end:
                return MemoryTimeQuery("unresolved_time", expression=text)
            return MemoryTimeQuery(mode, start.isoformat(), end.isoformat(), text[between.start():second.end()])
        match, (start, _) = spans[0]
        return MemoryTimeQuery(mode, EARLIEST, start.isoformat(), text[before.start():match.end()])
    if len(spans) > 1:
        return MemoryTimeQuery("unresolved_time", expression=text)
    if not spans:
        return MemoryTimeQuery("current", now.astimezone(UTC).isoformat())
    match, (start, end) = spans[0]
    expression = match[0]
    if (mode == "valid_range" and expression.lower() in {"today", "tomorrow"}
            and not re.search(r"\b(as of|valid|effective|held|was|were)\b", text, re.I)):
        # "What tea should I bring later today?" asks for a current
        # preference; it does not assert that an undated preference has a
        # certified historical validity interval.
        return MemoryTimeQuery("current", now.astimezone(UTC).isoformat())
    if re.search(r"\bsince\s+" + re.escape(expression), text, re.I):
        mode, end = "observed_range", now.astimezone(UTC)
    return MemoryTimeQuery(mode, start.isoformat(), end.isoformat(), expression)


def filter_unstructured(rows: list[dict], query: MemoryTimeQuery) -> list[dict]:
    """Observation questions cannot use write time as evidence of occurrence.

    A window the caller supplied never removes a quotation: in an event window,
    a quotation from outside it is kept and labelled ``outside_requested_window``,
    one with no occurrence time ``unknown``."""
    if query.mode == "current":
        return rows
    if query.mode == "observed_range":
        if query.supplied:
            return [dict(row, validity_status="source_occurrence_only" if query.accepts_observation(row.get("occurred_at"))
                         else "unknown" if utc_timestamp(row.get("occurred_at")) is None
                         else "outside_requested_window") for row in rows]
        return [dict(row, validity_status="source_occurrence_only")
                for row in rows if query.accepts_observation(row.get("occurred_at"))]
    # Source quotations alone do not establish when the asserted fact held.
    return [dict(row, validity_status="unknown") for row in rows]
