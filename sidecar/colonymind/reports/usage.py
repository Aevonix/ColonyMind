"""``colonymind report usage``: the model calls the instance made over a window, by lane, in counts only.

Read from the instance's ``usage.db`` (read-only; ``colonymind.usage``): calls, failures by outcome, input and output
tokens and p50/p90 latency per task and role and per endpoint, the last 24 hours' totals, and what the memory lanes
cost per captured message: the calls that turn a captured message into memory (its claims and their review, its
commitments, its appraisal, a description of its media and the embedding of its text), over the user messages of
the ordinary turns captured in the same window. Input and output tokens are estimated apart: a call whose endpoint
did not report one of them makes that side's estimate ``partial`` (a lower bound) or, when no memory call reported
it, ``unknown``; never a zero. ``--json`` gives the same numbers as sorted JSON with a fixed shape. Nothing is
written.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from colonymind import usage
from colonymind.util.temporal import now_utc

from .text import Document, plural

#: The tasks that turn a captured message into memory (the router's names, and the embedder's for a captured
#: source's text). A warmup or health check, a live query, a re-index and any other text are not among them.
MEMORY_TASKS = ("source_claim_extraction", "source_claim_review", "commitment_extract", "source_appraisal",
                "source_appraisal_revision", "source_image_description", "source_video_description",
                "source_embedding")


@dataclass
class Collected:
    span: str
    start: datetime
    end: datetime
    cut: bool
    found: bool
    totals: Dict[str, int]
    by_task: List[Dict[str, Any]]
    by_endpoint: List[Dict[str, Any]]
    memory: Dict[str, Any]
    last_day: Dict[str, int]


def _percentile(values: List[int], share: float) -> int:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    return round(statistics.quantiles(ordered, n=100, method="inclusive")[int(share * 100) - 1])


def _line(found: List[Mapping[str, Any]], **key: str) -> Dict[str, Any]:
    outcomes: Dict[str, int] = {}
    for row in found:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    latencies = [row["latency_ms"] for row in found]
    return {**key, **usage.totals(found), "outcomes": dict(sorted(outcomes.items())),
            "latency_ms": {"p50": _percentile(latencies, 0.5), "p90": _percentile(latencies, 0.9)}}


def _grouped(found: List[Mapping[str, Any]], fields: tuple) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in found:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    lines = [_line(rows, **dict(zip(fields, key))) for key, rows in groups.items()]
    return sorted(lines, key=lambda line: (-line["calls"], *(line[field] for field in fields)))


def _certainty(calls: int, missing: int) -> str:
    """How much of one side's token count is known: ``reported``, ``partial`` (a lower bound) or ``unknown``."""
    return "unknown" if calls and missing == calls else "partial" if missing else "reported"


def captured_messages(home: Path, since: datetime, until: Optional[datetime] = None) -> int:
    """The user messages captured in the window, read-only: those of the ordinary captured turns (person-scoped
    sources; a checkpoint's whole transcript and the mind's own records are session-scoped and not counted), as
    counted when each was captured (``usage.captured``), so a later erasure of the source never changes them."""
    return usage.captured_since(Path(home), since, until)


def collect(home: Path, *, since: str = "7d", now: Optional[datetime] = None) -> Collected:
    from .faculties import window
    now = now or now_utc()
    start, end, span = window(since, now=now)
    kept = now - usage.RETENTION
    cut = start < kept
    if cut:
        start, span = kept, f"the {usage.RETENTION.days} days usage is kept"
    home = Path(home)
    found = [row for row in usage.rows(home, since=start) if row["at"] <= usage.stamp(end)]
    memory_rows = [row for row in found if row["task"] in MEMORY_TASKS]
    captured = captured_messages(home, start, end)
    spent = usage.totals(memory_rows)
    per_message = None
    if captured:
        # A count an endpoint did not report is unknown, never zero, and each side on its own: a side is ``unknown``
        # when no memory call reported it, and ``partial`` (a lower bound) when only some did.
        tokens = {side: _certainty(spent["calls"], spent[f"without_{side}_tokens"]) for side in ("input", "output")}
        per_message = {"calls": round(spent["calls"] / captured, 2), "tokens": tokens,
                       **{f"{side}_tokens": None if tokens[side] == "unknown"
                          else round(spent[f"{side}_tokens"] / captured, 2) for side in ("input", "output")}}
    memory = {"captured_messages": captured, **{key: spent[key] for key in spent if key != "failed"},
              "per_message": per_message, "tasks": list(MEMORY_TASKS)}
    return Collected(span=span, start=start, end=end, cut=cut, found=(home / usage.DB_FILE).is_file(),
                     totals=usage.totals(found), by_task=_grouped(found, ("task", "role", "client")),
                     by_endpoint=_grouped(found, ("endpoint",)), memory=memory,
                     last_day=usage.last_day(home, now=now))


def to_json(collected: Collected) -> str:
    """The report as JSON: sorted keys, a fixed shape, the same numbers for the same rows."""
    data = {"window": {"span": collected.span, "since": usage.stamp(collected.start),
                       "until": usage.stamp(collected.end), "cut": collected.cut},
            "found": collected.found, "totals": collected.totals, "by_task": collected.by_task,
            "by_endpoint": collected.by_endpoint, "memory": collected.memory, "last_day": collected.last_day}
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def _counts(line: Mapping[str, Any]) -> str:
    text = f"{plural(line['calls'], 'call')}"
    failed = {outcome: count for outcome, count in line["outcomes"].items() if outcome != "ok"}
    if failed:
        text += " (" + ", ".join(f"{count} {outcome}" for outcome, count in failed.items()) + ")"
    text += f", {usage.tokens_in_words(line)}"
    return text + f"; latency p50 {line['latency_ms']['p50']:,} ms, p90 {line['latency_ms']['p90']:,} ms"


def document(collected: Collected) -> Document:
    doc = Document("Model usage")
    totals = collected.totals
    intro = [f"This covers {collected.span} to {collected.end.strftime('%Y-%m-%d %H:%M UTC')}: "
             f"{plural(totals['calls'], 'model call')}, {totals['failed']:,} failed; "
             f"{usage.tokens_in_words(totals)}."]
    if collected.cut:
        intro.append(f"Usage is kept {usage.RETENTION.days} days, so the window starts there.")
    if not collected.found:
        intro.append("No usage was recorded in this instance yet, so every count is zero.")
    intro.append("Counts only: no words of a prompt or an answer are kept.")
    doc.paragraph(" ".join(intro))
    doc.paragraph(f"Last 24 hours: {usage.describe(collected.last_day)}.")
    doc.heading("By task and role")
    doc.bullets([f"{line['task']} ({line['role']}, {line['client']}): {_counts(line)}" for line in collected.by_task]
                or ["no model calls in this window"])
    doc.heading("By endpoint")
    doc.bullets([f"{line['endpoint']}: {_counts(line)}" for line in collected.by_endpoint]
                or ["no model calls in this window"])
    doc.heading("Memory per captured message")
    memory = collected.memory
    spent = (f"The memory lanes (claims and their review, commitments, appraisals, media descriptions and the "
             f"embedding of a captured source's text) made {plural(memory['calls'], 'call')}")
    spent += f": {usage.tokens_in_words(memory)}." if memory["calls"] else "."
    per = memory["per_message"]
    if per is None:
        doc.paragraph(f"No messages were captured in this window, so there is no per-message cost. {spent}")
        return doc
    estimate = f"Per captured message: {per['calls']:g} {'call' if per['calls'] == 1 else 'calls'}"
    if per["tokens"] == {"input": "unknown", "output": "unknown"}:
        estimate += "; its token counts are unknown, since no memory call reported one"
    elif memory["calls"]:
        # The same words as the totals, each side's count divided by the messages captured.
        estimate += ", " + usage.tokens_in_words({**memory, "input_tokens": round(per["input_tokens"] or 0),
                                                  "output_tokens": round(per["output_tokens"] or 0)})
    estimate += "."
    doc.paragraph(f"{plural(memory['captured_messages'], 'user message')} captured in this window. {spent} {estimate}")
    return doc


__all__ = ["Collected", "MEMORY_TASKS", "captured_messages", "collect", "document", "to_json"]
