"""Read the verified-work projection without creating or repairing a database."""
from __future__ import annotations

from .text import plural


STATUSES = ("proven", "partial", "unproven", "failed")


def faculty(reading, section):
    if not section.on:
        section.records = False
        return
    from colonymind.work.store import read_summary

    result = read_summary(reading.instance.home / "colonymind-work.db", contact_id=reading.owner,
                          since=reading.start.isoformat(), until=reading.end.isoformat())
    total = int(result.get("total", 0))
    section.activity = total
    section.records = total > 0
    if not total:
        section.lines.append("No owner work records in this window.")
        return
    days = (reading.end - reading.start).total_seconds() / 86400
    counts = result.get("counts", {})
    section.lines.append(f"{plural(total, 'owner request')}; {total / days:.1f} roots per day.")
    section.lines.append("Status rates: " + ", ".join(
        f"{status} {int(counts.get(status, 0)) / total:.1%}" for status in STATUSES) + ".")
    corrected = int(result.get("owner_corrected", 0))
    section.lines.append(f"Owner-corrected rate: {corrected / total:.1%}.")
    unknown = result.get("unknown_effects", [])
    if unknown:
        named = [f"{item['tool']} {item['action']}: {item['count']}".strip() for item in unknown[:100]]
        section.lines.append("Unknown effects: " + "; ".join(named) + ".")
        if len(unknown) > 100:
            section.lines.append("Only the first 100 unknown effect kinds are shown.")
    else:
        section.lines.append("Unknown effects: none.")
