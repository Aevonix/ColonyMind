"""One consistent, read-only snapshot of the ledger's cited claims, as atlas lines.

A line is one stored claim, exactly as recall reads it: ``SourceClaimProjection._rows`` over one
speaker's own person-scoped sources, so the same eligibility applies (a withdrawn attribution, a
message no longer in its source and a derived claim whose grounding was erased are all absent).
On top of that the atlas drops sources whose projections were erased, the mind's own entries and
contacts the owner deleted. Superseded, corrected and expired claims are kept, marked, as history.

Every read happens inside one SQLite read transaction on a ``mode=ro`` connection, so an erasure
running beside the build is either wholly in the snapshot or wholly after it (the erasure mark
records which), and the build can never write the ledger.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from colonymind.util.temporal import parse_utc_text as utc


class AtlasError(RuntimeError):
    """The atlas cannot be built or checked as asked; nothing unsafe was written."""


SELF_TURN_SQL = "s.session_id<>'mind' AND s.turn_id NOT LIKE 'mind:%'"
STATUSES = ("current", "upcoming", "disputed", "expired", "superseded", "corrected", "withdrawn")


@dataclass(frozen=True)
class Line:
    """One cited claim. ``status``: current, upcoming (its stated validity has not begun), disputed (an
    attributed correction is on record for its message, so, as in recall, it is never current),
    expired (its validity ended), superseded (a later change replaced it), corrected (an explicit
    correction retracted it) or withdrawn (its successor is no longer on record, so it is history
    without a link)."""
    id: str
    speaker: str
    turn_id: str
    message_hash: str
    subject_key: str
    subject: str
    predicate: str
    value: str
    evidence: str
    kind: str                       # fact | quote | episode
    said_at: Optional[str]          # when the source says it was said (occurred_at), else None
    recorded_at: str                # when the ledger recorded it
    valid_from: Optional[str]
    valid_to: Optional[str]
    event_at: Optional[str]
    status: str
    successor: Optional[str]
    annotated: bool
    transcript: bool
    span: Optional[Tuple[int, int]] = None
    # The end (exclusive) of an event known only to a period wider than a day ("in 2015"): ``event_at`` is then
    # the period's start, never a date it happened on.
    event_until: Optional[str] = None

    @property
    def when(self) -> str:
        return self.said_at or self.recorded_at


@dataclass
class Snapshot:
    lines: List[Line]
    names: Dict[str, str]                         # speaker contact id -> display name
    messages: Dict[Tuple[str, str], str]          # (turn id, message hash) -> message text
    erasure_mark: List[int]
    # (turn id, message hash) -> attributed corrections on record: (recorded at, correction), or
    # (None, None) for one whose own source was erased since (the reference outlives it).
    notes: Dict[Tuple[str, str], List[Tuple[Optional[str], Optional[str]]]] = field(default_factory=dict)
    omitted: int = 0                              # eligible lines past ``max_lines``, the oldest said


def open_read_only(path: Path) -> sqlite3.Connection:
    path = Path(path)
    if not path.is_file():
        raise AtlasError(f"no database at {path}")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA query_only=ON")
    return conn


def erasure_mark(conn: sqlite3.Connection) -> List[int]:
    """[last erasure sequence, withdrawn attributions]: either moving means a page may cite a source
    that is gone or no longer that person's."""
    try:
        source_head = (conn.execute("SELECT coalesce(max(sequence),0) FROM curated_memory_events "
            "WHERE operation='forget' AND memory_id LIKE 'sources:%'").fetchone()[0]
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='curated_memory_events'").fetchone() else 0)
        return [int(conn.execute("SELECT coalesce(max(sequence),0) FROM source_erasures").fetchone()[0]) + source_head,
                int(conn.execute("SELECT count(*) FROM source_attribution_invalidations").fetchone()[0])]
    except sqlite3.DatabaseError as error:
        raise AtlasError(f"the ledger has no erasure tables ({type(error).__name__}); start the sidecar once") from None


def people(contacts_path: Optional[Path]) -> Tuple[Dict[str, str], Set[str]]:
    """Display names, and the contacts the owner deleted. No contacts store: ids stand in for names."""
    names: Dict[str, str] = {}
    deleted: Set[str] = set()
    if not contacts_path or not Path(contacts_path).is_file():
        return names, deleted
    with closing(open_read_only(Path(contacts_path))) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(contacts)")}
        if "contact_id" not in columns:
            return names, deleted
        wanted = [column for column in ("display_name", "given_name", "family_name", "deleted_at") if column in columns]
        for row in conn.execute("SELECT contact_id" + "".join("," + column for column in wanted) + " FROM contacts"):
            row = dict(row)
            if row.get("deleted_at"):
                deleted.add(str(row["contact_id"]))
                continue
            name = (row.get("display_name") or " ".join(
                part for part in (row.get("given_name"), row.get("family_name")) if part) or "").strip()
            if name:
                names[str(row["contact_id"])] = " ".join(name.split())[:160]
    return names, deleted


def _message_text(message: Dict[str, Any]) -> Optional[str]:
    from colonymind.turns.audio import claim_message
    view = claim_message(message)
    content = view.get("content") if view else None
    return content if isinstance(content, str) else None


def _event(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """When the claim says it happened: ``(at, until)``. A point or a calendar day is its start alone; any wider
    resolved interval (a vague date kept as the period it names) also its end, exclusive, so it is shown as
    that span and never as its first day."""
    event = data.get("event_time") or {}
    if isinstance(event, dict) and event.get("status") == "resolved" and event.get("start"):
        wider = event.get("precision") != "calendar_day" and event.get("end_exclusive")
        return str(event["start"]), (str(event["end_exclusive"]) if wider else None)
    return (str(data["event_at"]) if data.get("event_at") else None), None


def _notes(conn: sqlite3.Connection) -> Dict[Tuple[str, str], List[Tuple[Optional[str], Optional[str]]]]:
    """Attributed corrections (``turns.source_annotations``) by the exact message they qualify."""
    notes: Dict[Tuple[str, str], List[Tuple[Optional[str], Optional[str]]]] = {}
    from colonymind.memory import curated
    for row in conn.execute(
            "SELECT a.annotation_source_id, a.target_source_id, h.value AS message_hash, s.messages_json, s.occurred_at "
            "FROM source_annotations a JOIN json_each(a.target_message_hashes_json) h "
            "LEFT JOIN turn_sources s ON s.turn_id=a.annotation_source_id "
            "ORDER BY s.occurred_at, a.annotation_source_id"):
        correction = None
        if row["messages_json"] and not curated.is_forgotten(conn, row['annotation_source_id']):
            try:
                content = json.loads(json.loads(row["messages_json"])[0]["content"])
                correction = str(content["correction"]) if content.get("correction") else None
            except (ValueError, LookupError, TypeError, AttributeError):
                correction = None
        entry = (row["occurred_at"], correction) if correction else (None, None)
        notes.setdefault((row["target_source_id"], row["message_hash"]), []).append(entry)
    return notes


def _span(data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    start, end = data.get("span_start"), data.get("span_end")
    return (start, end) if isinstance(start, int) and isinstance(end, int) and 0 <= start < end else None


def read_snapshot(ledger_path: Path, contacts_path: Optional[Path], *, now: datetime,
                  max_lines: int) -> Snapshot:
    from colonymind.memory import curated
    from types import SimpleNamespace
    return curated.fenced_read(SimpleNamespace(db_path=Path(ledger_path)),
        lambda: _read_snapshot(ledger_path, contacts_path, now=now, max_lines=max_lines),
        empty=Snapshot(lines=[], names={}, messages={}, erasure_mark=[]))


def _read_snapshot(ledger_path, contacts_path, *, now, max_lines):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns.idempotency import source_message_hash

    names, deleted = people(contacts_path)
    reader = SourceClaimProjection(None)       # _rows reads only the connection it is given
    raw: List[Tuple[str, Dict[str, Any]]] = []
    messages: Dict[Tuple[str, str], str] = {}
    with closing(open_read_only(ledger_path)) as conn:
        conn.execute("BEGIN")
        try:
            mark = erasure_mark(conn)
            speakers = [row[0] for row in conn.execute(
                "SELECT DISTINCT s.contact_id FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
                f"WHERE s.scope='person' AND {SELF_TURN_SQL} ORDER BY s.contact_id")]
            erased = {row[0] for row in conn.execute("SELECT DISTINCT turn_id FROM source_projection_erasures")}
            notes = _notes(conn)
            for speaker in speakers:
                if speaker in deleted:
                    continue
                # Every eligible row (LIMIT -1): the cap applies after the lines are ordered by when
                # they were said, never to the ledger's recording order.
                for row in reader._rows(conn, speaker, "", limit=-1):
                    if row["turn_id"] not in erased:
                        raw.append((speaker, row))
            turns = sorted({row["turn_id"] for _, row in raw})
            kept_turns: Set[str] = set()
            for start in range(0, len(turns), 500):
                chunk = turns[start:start + 500]
                for source in conn.execute(
                        "SELECT s.turn_id, s.session_id, s.messages_json FROM turn_sources s WHERE s.turn_id IN ("
                        + ",".join("?" for _ in chunk) + f") AND {SELF_TURN_SQL}", chunk):
                    kept_turns.add(source["turn_id"])
                    for message in json.loads(source["messages_json"]):
                        text = _message_text(message) if message.get("role") == "user" else None
                        if text is not None:
                            messages[(source["turn_id"], source_message_hash(source["session_id"], message))] = text
        except sqlite3.DatabaseError as error:
            raise AtlasError(f"the ledger could not be read ({type(error).__name__}: {error})") from None
        finally:
            if conn.in_transaction:
                conn.execute("COMMIT")

    by_id = {row["id"]: (speaker, row) for speaker, row in raw if row["turn_id"] in kept_turns}
    lines = []
    for claim_id, (speaker, row) in by_id.items():
        if (row["turn_id"], row["message_hash"]) not in messages:
            continue
        status, successor = "current", None
        if row.get("retracted_by"):
            status, successor = "corrected", row["retracted_by"]
        elif row.get("superseded_by"):
            status, successor = "superseded", row["superseded_by"]
        elif row.get("valid_to") and (utc(row["valid_to"]) or now) <= now:
            status = "expired"
        elif (row.get("validity_basis") != "assertion_time" and row.get("valid_from")
              and (utc(row["valid_from"]) or now) > now):
            status = "upcoming"
        annotated = (row["turn_id"], row["message_hash"]) in notes
        if annotated and status in ("current", "upcoming"):
            # Recall never takes an annotated message as a current basis; neither does the atlas.
            status = "disputed"
        representation = row.get("representation") or ""
        kind = "episode" if representation == "episode" else "quote" if representation in {"preference", "procedure"} \
            else "fact"
        lines.append(Line(
            id=claim_id, speaker=speaker, turn_id=row["turn_id"], message_hash=row["message_hash"],
            subject_key=str(row.get("subject_key") or ""), subject=str(row.get("subject") or ""),
            predicate=str(row.get("predicate") or ""), value=str(row.get("value") or ""),
            evidence=str(row.get("evidence") or ""), kind=kind,
            said_at=row.get("observed_at") or None, recorded_at=str(row.get("recorded_at") or ""),
            # A scalar's validity defaults to the moment it was said: only a stated start is news.
            valid_from=None if row.get("validity_basis") == "assertion_time" else row.get("valid_from"),
            valid_to=row.get("valid_to"), event_at=_event(row)[0], event_until=_event(row)[1],
            status=status, successor=successor,
            annotated=annotated,
            transcript=row.get("epistemic_state") == "derived_unverified",
            span=_span(row)))
    # A successor that is not itself on record (its source erased) leaves the old value history,
    # never current again: a deleted correction must not revive what it corrected.
    ids = {line.id for line in lines}
    lines = [line if not line.successor or line.successor in ids else
             replace(line, status="withdrawn", successor=None) for line in lines]
    lines.sort(key=lambda line: (utc(line.when) or now, line.recorded_at, line.id), reverse=True)
    omitted = max(0, len(lines) - max(0, max_lines))
    lines = lines[:len(lines) - omitted]
    used = {(line.turn_id, line.message_hash) for line in lines}
    return Snapshot(lines=lines, names=names, messages={key: text for key, text in messages.items() if key in used},
                    erasure_mark=mark, notes={key: value for key, value in notes.items() if key in used},
                    omitted=omitted)
