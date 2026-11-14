"""The date-window lane: a person's sources from the span a question names (docs/RECALL-HYBRID.md, Recall and time).

"What did we talk about last week" names a span of time, not words: the week's messages share no word with it, and
the keyword and vector lanes rank what matches words or meaning. A time filter over what they found cannot bring
back a message neither of them ranked. When the question's time resolves to a span (``MemoryTimeQuery.window``,
from its words or from a search's own ``window``), this lane reads the person's sources in that span from the ledger:

- a source is in the span when it occurred there (``coalesce(occurred_at, ingested_at)``, read through the index
  this module builds), and a message is when one of its claims' event times falls there (a message sent today about
  last Tuesday);
- its candidates are ordered by the question's other words matched within the span (the lexical lane's full-text
  query, bm25, kept to the span), then by the vector lane's rank for the in-span messages it already returned, then
  newest first;
- they are spread over ``BUCKETS`` equal parts of the span, round robin, each part in that order, and each part's
  newest messages and events are read on their own, so a busy day cannot fill the lane; at most ``LIMIT``.

The lane only adds candidates: the reranker and its cutoff judge them like any other, and the claim readers apply
the question's time as before. The scope is the keyword lanes' own: the person's sources and this session's
session-scoped ones, never a source whose attribution was invalidated.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import logging
import re
import time
from typing import Any, Sequence

logger = logging.getLogger(__name__)
UTC = timezone.utc

#: The index on each person's sources by occurrence time.
INDEX = "source_occurrence_time"
#: The indexes on claims by event time: an interval's start, a point's instant.
CLAIM_INDEXES = {"source_claim_event_start": "json_extract(data_json,'$.event_time.start')",
                 "source_claim_event_at": "json_extract(data_json,'$.event_at')"}
#: The most candidates the lane returns (my choice, like the vector lane's 15).
LIMIT = 15
#: The equal parts of the span the candidates are spread over (my choice).
BUCKETS = 8
#: How far before the span an event interval may start and still reach into it (my choice): the longest period a
#: vague date resolves to is a year (``source_time.source_event_period``).
CLAIM_REACH = timedelta(days=400)
#: The best keyword matches the lane ranks in the full-text table alone before it reads their sources (my choice); when
#: they hold too few in the scope and the span, it reads every match kept to them (``_Reader.keyword``).
SCAN_LIMIT = 2000

_TIME = "coalesce(s.occurred_at, s.ingested_at)"
_SCOPE = ("s.contact_id=? AND (s.scope='person' OR s.session_id=?) "
          "AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)")
# The same scope for sources read by id: the unary plus keeps the planner on the id, not on the person's index.
_SCOPE_BY_ID = _SCOPE.replace("s.contact_id=?", "+s.contact_id=?", 1)
#: The newest sources in a span (contact, session, start, end, limit): what the index serves.
NEWEST_SQL = (f"SELECT s.turn_id, {_TIME} AS at FROM turn_sources AS s "
              f"WHERE {_SCOPE} AND {_TIME} >= ? AND {_TIME} < ? ORDER BY {_TIME} DESC LIMIT ?")
_OLDEST_SQL = (f"SELECT s.turn_id, {_TIME} AS at FROM turn_sources AS s "
               f"WHERE {_SCOPE} AND {_TIME} >= ? AND {_TIME} < ? ORDER BY {_TIME}")
_START = "json_extract(data_json,'$.event_time.start')"
_AT = "json_extract(data_json,'$.event_at')"
_RESOLVED = (f"coalesce(json_extract(c.data_json,'$.event_time.status'),'')='resolved' "
             "AND json_extract(c.data_json,'$.event_time.start') IS NOT NULL "
             "AND json_extract(c.data_json,'$.event_time.end_exclusive') IS NOT NULL")
# Each reads its index newest events first with every check the person's scope needs (the source's person and
# session, an invalidated attribution, a retraction, an overlap) applied before a row counts, so claims the person
# cannot see never use up what the lane reads; the reader stops once it has enough messages (``_Reader.claimed``).
_CLAIMS = ("SELECT c.turn_id, c.message_hash, {expr} AS at FROM source_claims AS c CROSS JOIN turn_sources AS s "
           "ON s.turn_id = c.turn_id WHERE {expr} >= ? AND {expr} < ? AND " + _SCOPE + " AND c.retracted_by IS NULL "
           "AND {kind} ORDER BY {expr} {order}")
_INTERVAL = dict(expr=_START.replace("data_json", "c.data_json"),
                 kind=f"{_RESOLVED} AND json_extract(c.data_json,'$.event_time.end_exclusive') > ?")
_POINT = dict(expr=_AT.replace("data_json", "c.data_json"), kind=f"NOT ({_RESOLVED})")
#: Claims whose event interval starts in a part of a span and reaches into the span (part start, part end, contact,
#: session, span start), newest first.
CLAIM_INTERVAL_SQL = _CLAIMS.format(**_INTERVAL, order="DESC")
#: Claims whose event instant falls in a part of a span (part start, part end, contact, session), newest first.
CLAIM_POINT_SQL = _CLAIMS.format(**_POINT, order="DESC")
# The same, oldest first: where a span open to the past begins (``_Reader.parts``).
_FIRST_INTERVAL_SQL = _CLAIMS.format(**_INTERVAL, order="ASC")
_FIRST_POINT_SQL = _CLAIMS.format(**_POINT, order="ASC")
# Stamps are UTC ISO strings in two spellings (``+00:00`` and ``Z``, with or without fractions): as text they sort as
# instants except within one second, so SQL reads a second either side and each stamp is then checked exactly.
_SLACK = timedelta(seconds=1)
# The start of a span open to the past ("before X": ``source_time.EARLIEST``).
_OPEN = datetime(1, 1, 1, tzinfo=UTC)


def initialize(conn) -> None:
    """Create the indexes the lane reads (the ledger's initializer runs this, never a turn); log how long a build
    over existing rows took."""
    present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    started = time.monotonic()
    if INDEX not in present:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {INDEX} ON turn_sources(contact_id, coalesce(occurred_at, "
                     "ingested_at))")
    for name, expression in CLAIM_INDEXES.items():
        if name not in present:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON source_claims({expression})")
    if {INDEX, *CLAIM_INDEXES} - present:
        sources = conn.execute("SELECT count(*) FROM turn_sources").fetchone()[0]
        claims = conn.execute("SELECT count(*) FROM source_claims").fetchone()[0]
        if sources or claims:
            logger.info("date-window indexes: built over %d sources and %d claims in %.2f s", sources, claims,
                        time.monotonic() - started)


def window_hits(ledger, *, query: str, contact_id: str, session_id: str, time_query,
                vector_hits: Sequence[dict[str, Any]] = (), limit: int = LIMIT) -> list[dict[str, Any]]:
    """The lane's hits for ``time_query``'s span, best first, shaped as the keyword lanes' (``search_sources``);
    none without a span."""
    span = time_query.window() if time_query is not None else None
    if span is None or not contact_id:
        return []
    start, end = span
    with closing(ledger._connect()) as conn:
        reader = _Reader(conn, contact_id, session_id, start, end)
        bounds = reader.parts()
        if bounds is None:
            return []
        claimed = reader.claimed(bounds, limit)
        tiers = [reader.keyword(_topic(query, time_query), claimed, limit),
                 [hit for hit in vector_hits if reader.inside(hit, claimed)],
                 reader.recent(bounds, claimed, limit)]
        # A message found through its claim is placed at its event's time, not at when it was sent.
        return _spread(tiers, bounds, limit, claimed)


def outside_window(projection, hits: Sequence[dict[str, Any]], shown: Sequence[dict[str, Any]], *, contact_id: str,
                   session_id: str, time_query, now: datetime, **options) -> list[dict[str, Any]]:
    """What a caller's window (``time_query.supplied``) would hide of the messages the lanes found, kept and labelled.

    The window sets which claims are admitted, and the projection reads a message a claim covers through that claim
    (``SourceClaimProjection.prepare_context``): a found message whose live claim the window does not admit would
    leave nothing, or only its unclaimed words. Those messages are projected again in the current view (a question
    with no date), under every correction and retraction rule, and what that adds to ``shown`` is returned, labelled
    ``outside_requested_window`` when the claim's time is known and outside the window, ``unknown`` when the window
    cannot place it. A retracted or superseded claim is never a reason: its words stay out, as without a window."""
    from colonymind.memory import curated
    return curated.fenced_read(projection.ledger, lambda: _read_outside_window(projection, hits, shown, contact_id=contact_id, session_id=session_id, time_query=time_query, now=now, **options), empty=[])


def _read_outside_window(projection, hits: Sequence[dict[str, Any]], shown: Sequence[dict[str, Any]], *, contact_id: str,
                   session_id: str, time_query, now: datetime, **options) -> list[dict[str, Any]]:
    from colonymind.beliefs.source_time import MemoryTimeQuery
    found = {(hit.get("turn_id"), hit.get("source_message_hash")) for hit in hits if hit.get("source_message_hash")}
    if not found:
        return []
    with closing(projection.ledger._connect()) as conn:
        claims = projection._rows(conn, contact_id, session_id, turn_ids=sorted({turn for turn, _ in found}),
                                  message_hashes=sorted({digest for _, digest in found}), limit=512)
    hidden: dict[tuple[str, str], str] = {}
    for claim in claims:
        identity = (claim["turn_id"], claim["message_hash"])
        if (identity not in found or claim.get("retracted_by") or claim.get("superseded_by")
                or time_query.accepts_claim(claim)):
            continue
        label = "outside_requested_window" if _dated(claim, time_query.mode) else "unknown"
        if hidden.get(identity) != "outside_requested_window":
            hidden[identity] = label
    if not hidden:
        return []
    current = MemoryTimeQuery("current", now.astimezone(UTC).isoformat())
    _, rows = projection.prepare_context(
        [], [hit for hit in hits if (hit.get("turn_id"), hit.get("source_message_hash")) in hidden],
        contact_id=contact_id, session_id=session_id, time_query=current, **options)
    overall = "outside_requested_window" if "outside_requested_window" in hidden.values() else "unknown"
    shown_ids = {row.get("id") for row in shown}
    kept = []
    for row in rows:
        if row.get("id") in shown_ids:
            continue
        carried = {(row.get("source_turn_id"), row.get("source_message_hash"))} | {
            (turn, digest) for turn, digests in (row.get("_source_message_hashes") or {}).items()
            for digest in digests}
        labels = {hidden[identity] for identity in carried if identity in hidden}
        label = ("outside_requested_window" if "outside_requested_window" in labels else "unknown" if labels
                 else overall)
        kept.append(dict(row, validity_status=label))
    return kept


def _dated(claim: dict, mode: str) -> bool:
    """Whether ``claim`` has the time a window of ``mode`` compares: an event time, or a validity start."""
    from colonymind.beliefs.source_time import event_interval
    if mode == "observed_range":
        return event_interval(claim.get("event_time")) is not None or _stamp(claim.get("event_at")) is not None
    return _stamp(claim.get("valid_from")) is not None


def _topic(query: str, time_query) -> str:
    """The question's words less the date that named the span: the span says when, not what."""
    expression = getattr(time_query, "expression", None)
    if expression and not getattr(time_query, "supplied", False):
        return re.sub(r"(?:\bsince\s+)?" + re.escape(expression), " ", query, count=1, flags=re.I)
    return query


def _stamp(value) -> datetime | None:
    from colonymind.beliefs.source_time import utc_timestamp
    return utc_timestamp(value) if isinstance(value, str) else None


def _less(at: datetime, by: timedelta) -> datetime:
    """``at`` less ``by``, or the start of the calendar when that lies before it."""
    try:
        return at - by
    except OverflowError:
        return _OPEN


def _before(at: datetime, by: timedelta = _SLACK) -> str:
    """``at`` less ``by``, as a stamp."""
    return _less(at, by).isoformat()


def _after(at: datetime) -> str:
    try:
        return (at + _SLACK).isoformat()
    except OverflowError:
        return at.isoformat()


class _Reader:
    """The lane's reads for one span [start, end), in one connection and the person's scope; each source's messages
    are read once."""

    def __init__(self, conn, contact_id: str, session_id: str, start: datetime, end: datetime) -> None:
        self.conn, self.scope, self.start, self.end = conn, (contact_id, session_id), start, end
        self._sources: dict[str, Any] = {}
        self._chunked: dict[str, list[dict]] = {}

    def within(self, at: datetime | None) -> bool:
        return at is not None and self.start <= at < self.end

    def source(self, turn_id: str):
        if turn_id not in self._sources:
            self._sources[turn_id] = self.conn.execute(
                f"SELECT s.turn_id, s.contact_id, s.session_id, s.scope, s.messages_json, s.occurred_at, "
                f"s.ingested_at FROM turn_sources AS s WHERE s.turn_id=? AND {_SCOPE_BY_ID}",
                (turn_id, *self.scope)).fetchone()
        return self._sources[turn_id]

    def parts(self) -> list[datetime] | None:
        """The bounds of ``BUCKETS`` equal parts of the span, oldest first (``BUCKETS + 1`` instants). A span open to
        the past ("before X") begins where the earliest source or event the lane can find in it does (my choice):
        equal parts of all time would put every source in the last one. None when such a span holds nothing."""
        low = self.start
        if low <= _OPEN:
            found = [at for at in (self.edge(), self.first_event()) if at is not None]
            if not found:
                return None
            low = max(low, min(found))
        width = (self.end - low) / BUCKETS
        return [low + width * index for index in range(BUCKETS)] + [self.end]

    def edge(self) -> datetime | None:
        """When the oldest source in the span occurred."""
        found = self.sources(self.start, self.end, 1, oldest=True)
        return found[0][1] if found else None

    def first_event(self) -> datetime | None:
        """The earliest event in the span among the person's claims (an interval's start, an instant)."""
        found = []
        for sql, args in ((_FIRST_INTERVAL_SQL, (_before(self.start, CLAIM_REACH), self.end.isoformat(), *self.scope,
                                                 self.start.isoformat())),
                          (_FIRST_POINT_SQL, (self.start.isoformat(), self.end.isoformat(), *self.scope))):
            for row in self.conn.execute(sql, args):
                at = _stamp(row["at"])
                if at is not None:
                    found.append(at)
                    break
        return min(found) if found else None

    def sources(self, start: datetime, end: datetime, limit: int, *, oldest: bool = False) -> list:
        """The newest (or oldest) sources in [start, end) and the span, at most ``limit``: (source, when)."""
        start, end = max(start, self.start), min(end, self.end)
        found, offset, page = [], 0, max(limit, 16)
        sql = (_OLDEST_SQL + " LIMIT ?" if oldest else NEWEST_SQL) + " OFFSET ?"
        while start < end and len(found) < limit:
            rows = self.conn.execute(sql, (*self.scope, _before(start), _after(end), page, offset)).fetchall()
            for row in rows:
                at = _stamp(row["at"])
                if at is not None and start <= at < end:
                    found.append((row["turn_id"], at))
            if len(rows) < page:
                break
            offset += page
        return found[:limit]

    def claimed(self, bounds: list[datetime], limit: int) -> dict[tuple[str, str], datetime]:
        """Messages whose claims' event time falls in the span, read part by part (``bounds``): in each part at most
        ``limit`` messages, newest events first, so events crowded into one part never hide another's. (source,
        message hash) to its newest event's time, clamped to the span: an event period that began before the span
        is placed at its start, in the first part."""
        latest = self.end - timedelta(microseconds=1)
        found: dict[tuple[str, str], datetime] = {}
        for index in reversed(range(BUCKETS)):
            low, high = bounds[index], bounds[index + 1]
            # A second either side, then each event is checked exactly (``_SLACK``); the first part reaches back to
            # periods that began before the span.
            reach = _before(low, CLAIM_REACH) if index == 0 else _before(low)
            first = _less(low, CLAIM_REACH) if index == 0 else low
            part: dict[tuple[str, str], datetime] = {}
            for sql, args in ((CLAIM_INTERVAL_SQL, (reach, _after(high), *self.scope, self.start.isoformat())),
                              (CLAIM_POINT_SQL, (_before(low), _after(high), *self.scope))):
                messages: set[tuple[str, str]] = set()
                for row in self.conn.execute(sql, args):
                    at = _stamp(row["at"])
                    identity = (row["turn_id"], row["message_hash"])
                    if (at is None or identity in messages or at >= high
                            or at < (first if sql is CLAIM_INTERVAL_SQL else low)):
                        continue
                    messages.add(identity)
                    part.setdefault(identity, min(max(at, bounds[0]), latest))
                    if len(messages) == limit:
                        break
            for identity, at in sorted(part.items(), key=lambda item: item[1], reverse=True)[:limit]:
                found.setdefault(identity, at)
        return found

    def inside(self, hit: dict, claimed) -> bool:
        return (self.within(_stamp(hit.get("occurred_at") or hit.get("ingested_at")))
                or (hit.get("turn_id"), hit.get("source_message_hash")) in claimed)

    def keyword(self, text: str, claimed, limit: int) -> list[dict]:
        """The in-span chunks the question's words match, bm25 first: the lexical lane's full-text query kept to the
        person's scope and to the span's sources and the messages found through their claims. As the keyword lane
        reads (``idempotency._scoped_matches``), the best ``SCAN_LIMIT`` matches are ranked in the full-text table
        alone and only their sources are read; when they hold fewer than ``limit`` hits and more matched, the query
        runs over every match kept to the scope and the span, so better matches outside them never hide these."""
        from colonymind.turns import keywords
        expression = keywords.match_expression(keywords.query_terms(self.conn, text))
        if expression is None:
            return []
        by_source: dict[str, set[str]] = {}
        for turn, digest in claimed:
            by_source.setdefault(turn, set()).add(digest)
        top = self.conn.execute("SELECT rowid, turn_id, role, content FROM turn_source_search WHERE "
                                "turn_source_search MATCH ? ORDER BY bm25(turn_source_search), rowid LIMIT ?",
                                (expression, SCAN_LIMIT + 1)).fetchall()
        first = top[:SCAN_LIMIT]
        turns = list(dict.fromkeys(row["turn_id"] for row in first))
        inside = set()
        for index in range(0, len(turns), 500):
            group = turns[index:index + 500]
            inside.update(row["turn_id"] for row in self.conn.execute(
                f"SELECT s.turn_id, {_TIME} AS at FROM turn_sources AS s WHERE s.turn_id IN "
                f"({','.join('?' * len(group))}) AND {_SCOPE_BY_ID}", (*group, *self.scope))
                if self.within(_stamp(row["at"])))
        hits: list[dict] = []
        for row in first:
            if self._matched(row, row["turn_id"] in inside, by_source, hits) >= limit:
                return hits
        if len(top) <= SCAN_LIMIT:
            return hits
        read = {row["rowid"] for row in first}
        claimed_turns = sorted(by_source)
        rows = self.conn.execute(
            f"SELECT f.rowid AS rowid, f.turn_id, f.role, f.content, {_TIME} AS at FROM turn_source_search AS f "
            f"JOIN turn_sources AS s ON s.turn_id=f.turn_id WHERE turn_source_search MATCH ? AND {_SCOPE} "
            f"AND (({_TIME} >= ? AND {_TIME} < ?) OR s.turn_id IN ({','.join('?' * len(claimed_turns))})) "
            f"ORDER BY bm25(turn_source_search), f.rowid",
            (expression, *self.scope, _before(self.start), _after(self.end), *claimed_turns))
        for row in rows:
            if row["rowid"] not in read and self._matched(row, self.within(_stamp(row["at"])), by_source,
                                                          hits) >= limit:
                break
        return hits

    def _matched(self, row, in_span: bool, by_source: dict[str, set[str]], hits: list[dict]) -> int:
        """Add the chunks a full-text match names to ``hits`` when its source is in the span or its message was
        found through a claim; the count of hits."""
        claims = by_source.get(row["turn_id"], set())
        if not in_span and not claims:
            return len(hits)
        source = self.source(row["turn_id"])
        if source is not None:
            hits.extend(hit for hit in self._chunks(source) if hit["role"] == row["role"]
                        and hit["content"] == row["content"] and (in_span or hit["source_message_hash"] in claims))
        return len(hits)

    def recent(self, bounds: list[datetime], claimed, limit: int) -> list[dict]:
        """Each part's newest messages (from the span's sources, and the messages found through their claims),
        newest first; a message stands in by its first chunk."""
        entries = []
        for index in range(BUCKETS):
            part_start, part_end = bounds[index], bounds[index + 1]
            part = [(turn, None, at) for turn, at in self.sources(part_start, part_end, limit)]
            part += [(turn, digest, at) for (turn, digest), at in claimed.items() if part_start <= at < part_end]
            entries += sorted(part, key=lambda entry: entry[2], reverse=True)[:limit]
        entries.sort(key=lambda entry: entry[2], reverse=True)
        hits = []
        for turn, digest, _ in entries:
            source = self.source(turn)
            if source is None:
                continue
            first = {}
            for hit in self._chunks(source):
                if (digest is None or hit["source_message_hash"] == digest) and hit["source_message_hash"] not in first:
                    first[hit["source_message_hash"]] = hit
            hits.extend(first.values())
        return hits

    def _chunks(self, source) -> list[dict]:
        """Every chunk of ``source``, shaped as ``search_sources`` returns a hit."""
        if source["turn_id"] not in self._chunked:
            self._chunked[source["turn_id"]] = _read_chunks(source)
        return self._chunked[source["turn_id"]]


def _read_chunks(source) -> list[dict]:
    from colonymind.turns.audio import evidence_metadata
    from colonymind.turns.idempotency import _lexical_chunks, source_message_hash
    try:
        messages = json.loads(source["messages_json"])
    except (TypeError, ValueError):
        return []
    common = {name: source[name] for name in ("turn_id", "contact_id", "session_id", "scope", "occurred_at",
                                              "ingested_at")}
    chunks, identities = [], {}
    for message, text, chunk in _lexical_chunks(messages if isinstance(messages, list) else []):
        key = id(message)
        if key not in identities:
            identities[key] = (source_message_hash(source["session_id"], message), evidence_metadata(message))
        digest, metadata = identities[key]
        chunks.append({**common, "role": message.get("role"), "content": chunk, "source_message_hash": digest,
                       **metadata, **({"excerpt_truncated": True} if chunk != text else {})})
    return chunks


def _spread(tiers: Sequence[Sequence[dict]], bounds: list[datetime], limit: int,
            claimed: dict[tuple[str, str], datetime]) -> list[dict]:
    """The tiers' hits in order (each message once), spread round robin over the ``BUCKETS`` parts of ``bounds`` by
    their time (the event's for a message found through its claim); each round takes the next of every part, the
    part holding the best of them first."""
    parts: dict[int, list[tuple[int, dict]]] = {}
    seen, position = set(), 0
    for tier in tiers:
        for hit in tier:
            identity = (hit.get("turn_id"), hit.get("source_message_hash"))
            if identity in seen:
                continue
            seen.add(identity)
            at = claimed.get(identity) or _stamp(hit.get("occurred_at") or hit.get("ingested_at"))
            index = min(BUCKETS - 1, max(0, bisect_right(bounds, at) - 1)) if at is not None else 0
            parts.setdefault(index, []).append((position, hit))
            position += 1
    chosen = []
    while parts and len(chosen) < limit:
        for index in sorted(parts, key=lambda index: parts[index][0][0]):
            chosen.append(parts[index].pop(0)[1])
            if not parts[index]:
                del parts[index]
            if len(chosen) == limit:
                break
    return chosen


__all__ = ["BUCKETS", "CLAIM_INDEXES", "CLAIM_INTERVAL_SQL", "CLAIM_POINT_SQL", "INDEX", "LIMIT",
           "NEWEST_SQL", "SCAN_LIMIT", "initialize", "outside_window", "window_hits"]
