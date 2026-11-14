"""``colonymind report claims``: the claims formed from the messages captured over a window, in counts only.

Read from the instance's ``turn-idempotency.db`` (read-only, ``reports.instance``): the person-scoped sources captured
in the window (``turn_sources.ingested_at``) and their extraction jobs, the claims formed from them by extraction
version, and the claims whose subject an earlier message of the conversation named (``subject_basis_message``,
docs/SOURCE-CLAIMS.md): how many were formed, what became of each (it stands; it is hidden because the message that
named its subject was annotated since; a correction retracted it; a later change replaced it), and the proposals the
extraction refused for their basis, by reason (each job's own diagnostics). A claim erased with its message, or with
the message that named its subject, is gone and not counted. That feature is measured live: this is its report line,
and the counts by extraction version set a build beside the one before it. Counts only: no words, names or ids.
``--json`` gives the same numbers as sorted JSON with a fixed shape. Nothing is written.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from colonymind.util.temporal import now_utc

from .instance import LEDGER, Instance
from .text import Document, listing, percent, plural

#: The extraction's refusals of a subject named in an earlier message (``source_claims.message_basis`` and the
#: commit's check): every rejection reason with this prefix.
BASIS_REFUSAL = "subject_basis"


class Unreadable(Exception):
    """The ledger exists but cannot be read (damaged, or this user may not open it): its counts are unknown."""


@dataclass
class Collected:
    span: str
    start: datetime
    end: datetime
    found: bool
    unreadable: bool
    sources: Dict[str, Any]
    claims: Dict[str, Any]
    earlier: Dict[str, Any]


def _stamp(at: datetime) -> str:
    """``at`` as the ledger stamps ``ingested_at`` (``strftime('%Y-%m-%dT%H:%M:%fZ')``), so the two compare."""
    return at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


_CAPTURED = "s.scope='person' AND s.ingested_at>=? AND s.ingested_at<?"


def _basis_stands_sql() -> str:
    from colonymind.beliefs.source_projection import _message_basis_source_sql
    basis = "json_extract(c.data_json,'$.subject_basis_message.{}')".format
    return "EXISTS (SELECT 1 " + _message_basis_source_sql(basis("turn_id"), "s.contact_id",
                                                           basis("message_hash")) + ")"


def collect(home: Any, *, since: str = "7d", now: Optional[datetime] = None) -> Collected:
    from .faculties import window
    now = now or now_utc()
    start, end, span = window(since, now=now)
    instance = Instance(Path(home))
    window_params = (_stamp(start), _stamp(end))
    try:
        jobs = instance.query(LEDGER, "SELECT j.status, j.diagnostics_json FROM source_claim_jobs j JOIN turn_sources "
                                      "s ON s.turn_id=j.turn_id WHERE " + _CAPTURED, window_params)
        formed = instance.query(LEDGER, "SELECT coalesce(json_extract(c.data_json,'$.extraction_version'),'unknown') "
                                        "AS version, json_extract(c.data_json,'$.subject_basis_message') IS NOT NULL "
                                        "AS named_earlier, c.retracted_by IS NOT NULL AS corrected, c.superseded_by "
                                        "IS NOT NULL AS replaced FROM source_claims c JOIN turn_sources s ON "
                                        "s.turn_id=c.turn_id WHERE " + _CAPTURED, window_params)
        # Whether the message that named each one's subject still names it, read apart: a store from before the
        # tables it reads says nothing about it (unknown, never "none hidden").
        stands = instance.query(LEDGER, "SELECT " + _basis_stands_sql() + " AS stands FROM source_claims c JOIN "
                                "turn_sources s ON s.turn_id=c.turn_id WHERE " + _CAPTURED + " AND c.retracted_by IS "
                                "NULL AND c.superseded_by IS NULL AND json_extract(c.data_json,"
                                "'$.subject_basis_message') IS NOT NULL", window_params)
        found = instance.db(LEDGER) is not None
    finally:
        instance.close()
    if LEDGER in instance.unreadable:
        raise Unreadable(f"{Path(home) / LEDGER} cannot be read, so what the extractor formed is unknown")
    by_status: Dict[str, int] = {}
    refused: Dict[str, int] = {}
    for row in jobs:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        try:
            counts = (json.loads(row["diagnostics_json"]) or {}).get("rejection_counts") or {}
        except (TypeError, ValueError, AttributeError):
            counts = {}
        for reason, count in counts.items() if isinstance(counts, dict) else ():
            if str(reason).startswith(BASIS_REFUSAL) and isinstance(count, int):
                refused[reason] = refused.get(reason, 0) + count
    by_version: Dict[str, int] = {}
    for row in formed:
        by_version[row["version"]] = by_version.get(row["version"], 0) + 1
    earlier = [row for row in formed if row["named_earlier"]]
    corrected = sum(1 for row in earlier if row["corrected"])
    replaced = sum(1 for row in earlier if row["replaced"] and not row["corrected"])
    current = len(earlier) - corrected - replaced
    known = len(stands) == current
    hidden = sum(1 for row in stands if not row["stands"]) if known else None
    return Collected(
        span=span, start=start, end=end, found=found, unreadable=False,
        sources={"captured": len(jobs), "by_status": dict(sorted(by_status.items()))},
        claims={"formed": len(formed), "by_version": dict(sorted(by_version.items()))},
        earlier={"formed": len(earlier), "standing": None if hidden is None else current - hidden, "hidden": hidden,
                 "corrected": corrected, "replaced": replaced,
                 "refused": {"total": sum(refused.values()), "by_reason": dict(sorted(refused.items()))}})


def to_json(collected: Collected) -> str:
    """The report as JSON: sorted keys, a fixed shape, the same numbers for the same store."""
    data = {"window": {"span": collected.span, "since": _stamp(collected.start), "until": _stamp(collected.end)},
            "found": collected.found, "sources": collected.sources, "claims": collected.claims,
            "earlier_message_subjects": collected.earlier}
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def _were(count: int) -> str:
    return "was" if count == 1 else "were"


def document(collected: Collected) -> Document:
    doc = Document("Claims formed")
    claims, earlier = collected.claims, collected.earlier
    intro = [f"This covers {collected.span} to {collected.end.strftime('%Y-%m-%d %H:%M UTC')}, by when each message "
             f"was captured: {plural(collected.sources['captured'], 'captured source')} with a claim job."]
    if not collected.found:
        intro.append("No ledger was found in this instance yet, so nothing was captured.")
    intro.append("Counts only: no words, names or ids of what memory holds.")
    doc.paragraph(" ".join(intro))
    if not claims["formed"]:
        doc.paragraph("No claim was formed from the messages captured in this window.")
    else:
        doc.paragraph(f"{plural(claims['formed'], 'claim')} {_were(claims['formed'])} formed, by extraction version:")
        doc.bullets([f"{version}: {plural(count, 'claim')}" for version, count in claims["by_version"].items()])
    doc.heading("Subjects named in an earlier message")
    doc.paragraph("A claim whose message names its subject only by a pronoun may take the subject from an exact span "
                  "of an earlier message of the same conversation (extraction version 19). A correction of one can mean "
                  "it named the wrong subject.")
    lines = []
    if earlier["formed"]:
        fate = []
        if earlier["standing"] is not None:
            fate.append(f"{earlier['standing']} {'stands' if earlier['standing'] == 1 else 'stand'}")
            fate.append(f"{earlier['hidden']} {'is' if earlier['hidden'] == 1 else 'are'} hidden (the message that "
                        "named the subject was annotated since)")
        else:
            fate.append("whether the rest stand is unknown (this store keeps no annotations yet)")
        fate.append(f"{earlier['corrected']} {_were(earlier['corrected'])} retracted by a correction")
        fate.append(f"{earlier['replaced']} {_were(earlier['replaced'])} replaced by a later change")
        lines.append(f"{plural(earlier['formed'], 'claim')} took {'its' if earlier['formed'] == 1 else 'their'} "
                     f"subject from an earlier message ({percent(earlier['formed'], claims['formed'])} of the claims "
                     f"formed): {listing(fate)}.")
    else:
        lines.append("No claim formed in this window took its subject from an earlier message.")
    refused = earlier["refused"]
    if refused["total"]:
        reasons = listing([f"{reason} {count}" for reason, count in refused["by_reason"].items()])
        lines.append(f"The extraction refused {plural(refused['total'], 'proposal')} for the earlier message it "
                     f"named: {reasons}.")
    else:
        lines.append("The extraction refused no proposal for the earlier message it named.")
    lines.append("A claim erased with its message, or with the message that named its subject, is gone and not "
                 "counted.")
    doc.bullets(lines)
    return doc


__all__ = ["BASIS_REFUSAL", "Collected", "Unreadable", "collect", "document", "to_json"]
