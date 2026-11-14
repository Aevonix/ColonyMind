"""Outside observations: surprises, merged concerns and sent messages per source, read-only.

An absent or stale source is quiet. It does not turn an otherwise healthy faculty into a report warning.
The observation reader supplies counts only; unverified descriptions and images never enter this report.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .instance import INITIATIVES, MIND, Instance, loads, parse_time
from .text import plural

if TYPE_CHECKING:
    from .faculties import Reading, Section


def collect(instance: Instance, *, start: datetime, end: datetime) -> list[dict]:
    """Counts per source, with messages measured at confirmed delivery in ``[start, end)``.

    Mind-state counters record observations and concerns. Only a sent intention establishes delivery;
    an observation's request to tell, or a message the gate held, contributes no message count.
    """
    from colonymind.mind.observations import report_rows

    observations = instance.read(MIND, lambda conn: report_rows(conn, start=start, end=end), [])
    counts = {row["source"]: {**row, "messages": 0} for row in observations}
    delivered = instance.query(
        INITIATIVES, "SELECT context, completed_at FROM initiatives "
                    "WHERE kind = ? AND type = ? AND status = ?",
        ("message", "percept_tell", "sent"),
    )
    for message in delivered:
        at = parse_time(message["completed_at"])
        if at is None or not start <= at < end:
            continue
        source = loads(message["context"], {}).get("observation_source")
        if not isinstance(source, str) or not source:
            continue
        row = counts.setdefault(source, {"source": source, "surprises": 0, "concerns": 0,
                                         "messages": 0, "ignored": 0})
        row["messages"] += 1
    return [counts[source] for source in sorted(counts)]


def read(reading: Reading, section: Section) -> None:
    """Fill the faculty section without opening a writable observation or intention store."""
    rows = collect(reading.instance, start=reading.start, end=reading.end)
    if not section.on:
        section.lines.append("With senses off, posts are accepted and ignored.")
    if not rows:
        section.records = False
        section.lines.append("No outside sources posted observations in this window.")
        return
    for row in sorted(rows, key=lambda item: item["source"]):
        surprises, concerns, messages = (row.get(key, 0) for key in ("surprises", "concerns", "messages"))
        line = (f"{row['source']}: {plural(surprises, 'surprise')}, {plural(concerns, 'concern')} and "
                f"{plural(messages, 'message')}")
        ignored = row.get("ignored", 0)
        if ignored:
            line += f"; {plural(ignored, 'observation')} ignored"
        section.lines.append(line + ".")
        section.activity += surprises + concerns + messages
    # Accepted-but-ignored posts, including ones beyond the cap, are not activity while the faculty is off.
    section.records = section.activity > 0


__all__ = ["collect", "read"]
