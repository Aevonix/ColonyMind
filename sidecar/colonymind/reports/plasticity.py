"""Plasticity's learner lines: teacher evidence, live comparison, changes and erasures."""
from __future__ import annotations

import json
from copy import deepcopy
from importlib import import_module
from typing import Any, Mapping

from .text import Document, plural

SETTING = "mind.faculties.plasticity"


def _learner_module(implementation):
    if not isinstance(implementation, str) or not implementation.isidentifier():
        return None
    name = f"colonymind.plasticity.learners.{implementation}"
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return None


def collect(config, *, runtime=None) -> dict:
    """Read the persisted report without constructing learners or changing instance files."""
    if runtime is None:
        from colonymind.plasticity.nightly import read_report
        result = read_report(config)
    else:
        result = runtime.report()
    config = config or getattr(runtime, "config", None)
    if config is None:
        return result
    for row in result.get("learners", []):
        declaration = config.get("plasticity.learners", {}).get(row["name"], {})
        implementation = declaration.get("implementation", row["name"])
        report = getattr(_learner_module(implementation), "report", None)
        if callable(report):
            row["implementation"] = implementation
            row["detail"] = report(config, row["name"], runtime=runtime)
            # Keep the implementation-named JSON detail available to existing report consumers.
            row.setdefault(implementation, row["detail"])
    return result


def comparator_line(result: Mapping[str, Any]) -> str:
    if result.get("kind") == "numeric-online":
        count = result.get("labelled", 0)
        text = (f"On {plural(count, 'labelled numeric outcome')}, relative excess loss against the previous "
                f"revision is {result['relative_excess']:+.2%}.") if count else "Awaiting numeric outcomes."
        previous = result if result.get("rolled_back") else result.get("last_rollback")
        if previous:
            text += " Rolled back: " + str(previous["reason"]).rstrip(".") + "."
        else:
            text += " " + str(result.get("reason", "")).rstrip(".") + "."
        health = result.get("health")
        if health is not None:
            text += (f" Streams: {health.get('streams', 0)}; forecasts served: {health.get('forecasts', 0)}; "
                     f"surprises: {health.get('surprises', 0)}; notes raised: {health.get('notes', 0)}; "
                     f"ignored observations: {health.get('ignored', 0)}.")
            leads = health.get("lead_seconds", [])
            text += (f" Mean lead over fixed thresholds: {sum(leads) / len(leads):g} seconds."
                     if leads else " No paired fixed-threshold warning yet.")
        return text
    count = result.get("labelled", 0)
    if not count:
        return "The live comparison has no labelled outcomes yet."
    text = (f"On {plural(count, 'labelled outcome')}, the serving revision lost {result.get('losses', 0)} "
            f"and gained {result.get('gains', 0)} against the previous revision on the same inputs.")
    if result.get("rolled_back"):
        text += " Rolled back: " + str(result.get("reason") or "measured regression") + "."
    elif result.get("reason"):
        text += " " + str(result["reason"]).rstrip(".") + "."
    return text


def learner_line(row: Mapping[str, Any]) -> str:
    events = row.get("events", [])
    counts = {name: sum(event.get("event") == name for event in events)
              for name in ("promoted", "dropped", "rolled_back")}
    erasures = row.get("erasures", 0)
    if isinstance(erasures, (list, tuple)):
        erasures = len(erasures)
    parts = [f"{row['name']}: {counts['promoted']} promoted, {counts['dropped']} refused, "
             f"{counts['rolled_back']} rolled back; {plural(erasures, 'erasure')}."]
    for teacher in row.get("teachers", []):
        proof = "counts as proof" if teacher["counts_as_proof"] else "does not count as proof"
        loss = f", loss {teacher['loss']:g}" if teacher.get("loss") is not None else ""
        parts.append(f"{teacher['teacher']}: {plural(teacher['rows'], 'row')}{loss}, {proof}.")
    if row.get("teacher_note"):
        parts.append(str(row["teacher_note"]).rstrip(".") + ".")
    reasons = list(dict.fromkeys(str(event["reason"]) for event in events if event.get("reason")))
    if reasons:
        parts.append("Revision decisions: " + "; ".join(reasons) + ".")
    parts.append(comparator_line(row.get("comparator", {})))
    if "detail" in row:
        report_lines = getattr(_learner_module(row.get("implementation")), "report_lines", None)
        if callable(report_lines):
            parts.extend(report_lines(row["detail"]))
    return " ".join(parts)


def lines(collected: Mapping[str, Any]) -> list[str]:
    result = []
    if not collected.get("enabled", True):
        result.append("Plasticity is switched off; every learner serves its base.")
    elif not collected.get("learners") and not collected.get("rejected"):
        result.append("No learners configured; plasticity is on and idle.")
    if collected.get("note"):
        result.append(str(collected["note"]))
    result.extend(learner_line(row) for row in collected.get("learners", []))
    result.extend(f"{name}: did not load; {reason}." for name, reason in collected.get("rejected", {}).items())
    return result


def document(collected: Mapping[str, Any]) -> Document:
    report = Document("Plasticity")
    report.bullets(lines(collected))
    report.paragraph(f"Setting: {SETTING}")
    return report


def to_json(collected: Mapping[str, Any]) -> str:
    return json.dumps(collected, sort_keys=True, allow_nan=False) + "\n"


def faculty(reading, section) -> None:
    """The faculty report dispatcher calls this with its existing instance configuration."""
    collected = deepcopy(collect(reading.instance.config))
    for row in collected.get("learners", []):
        row["events"] = [event for event in row.get("events", []) if reading.in_window(event.get("at"))]
    if collected.get("learners"):
        section.lines.append("Revision counts cover this window. Teachers describe the latest build; the live "
                             "comparison has its own rolling window, and erasures are lifetime totals.")
    section.lines.extend(lines(collected))
    section.records = bool(collected.get("learners") or collected.get("rejected"))
    section.activity = sum(len(row.get("events", [])) for row in collected.get("learners", []))


def _read_behaviour_snapshot(path, reader, *, attempts: int = 6, pause: float = 0.5):
    """Back up an isolated copy, never attaching SQLite to the source files.

    Even a read-only SQLite WAL connection can update the source -shm read marks.
    Copy the database and rollback journal, then the WAL last, under TMPDIR,
    omitting -shm so SQLite rebuilds that index locally. A live writer appends to
    the WAL during a long copy; that is accepted, because within one WAL generation
    the file only grows and SQLite recovers the copied prefix up to its last valid
    commit frame (frame checksums reject a torn tail). The copy is rejected and
    retried, with backoff, if the main database or journal changed (a checkpoint
    wrote pages) or the WAL changed generation (its header: checkpoint sequence and
    salts) or was replaced. Backup then resolves committed WAL frames into a
    standalone database. Both copies and all temporary side files are removed on
    success or failure.
    """
    import shutil
    import sqlite3
    import tempfile
    import time
    from contextlib import closing
    from pathlib import Path

    wal = path.with_name(path.name + "-wal")
    files = [path, path.with_name(path.name + "-journal"), wal]   # the WAL is copied last

    def signatures():
        result = []
        for source in files:
            try:
                stat = source.stat()
                if source == wal:
                    with source.open("rb") as stream:
                        header = stream.read(32)
                    result.append((stat.st_dev, stat.st_ino, header))   # may grow; generation must hold
                else:
                    result.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            except FileNotFoundError:
                result.append(None)
        return result

    for attempt in range(attempts):
        if attempt:
            time.sleep(min(pause * 2 ** (attempt - 1), 8.0))
        before = signatures()
        if before[0] is None:
            return None
        with tempfile.TemporaryDirectory(prefix="colonymind-behaviour-") as directory:
            directory = Path(directory)
            try:
                for source, signature in zip(files, before):
                    if signature is not None:
                        shutil.copyfile(source, directory / source.name)
            except FileNotFoundError:
                continue  # A writer removed/replaced a side file; try a fresh copy.
            if signatures() != before:
                continue
            # Only the disposable copy is opened with SQLite. It sees committed WAL
            # rows and may create/update its own -shm while making the backup.
            with closing(sqlite3.connect(directory / path.name)) as source, closing(
                    sqlite3.connect(directory / "snapshot.db")) as snapshot:
                source.backup(snapshot)
                snapshot.row_factory = sqlite3.Row
                snapshot.execute("PRAGMA query_only=ON")
                return reader(snapshot)
    raise ValueError("the behaviour ledger changed during snapshot; counts are unknown")


def collect_behaviour(home, *, since: str | None = None, until: str | None = None) -> dict:
    """Measure captured history only; dates include both named UTC days."""
    import re
    import sqlite3
    from datetime import date, datetime, time, timedelta, timezone
    from colonymind.plasticity.learners.behaviour import read_rows, reply_counts, tool_counts
    from .instance import Instance, LEDGER

    def boundary(value):
        if value is None:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("dates must use YYYY-MM-DD")
        try:
            return datetime.combine(date.fromisoformat(value), time(), timezone.utc)
        except ValueError:
            raise ValueError("dates must use a valid YYYY-MM-DD") from None

    start, last = boundary(since), boundary(until)
    if start is not None and last is not None and last < start:
        raise ValueError("--until must be on or after --since")
    try:
        end = last + timedelta(days=1) if last is not None else None
    except OverflowError:
        raise ValueError("--until must be before 9999-12-31") from None
    instance = Instance(home)

    def read(conn):
        conn.execute("BEGIN")
        try:
            return read_rows(conn, instance.owner_id, since=start, until=end)
        finally:
            conn.rollback()

    try:
        try:
            rows = _read_behaviour_snapshot(instance.home / LEDGER, read)
        except (sqlite3.Error, OSError):
            raise ValueError("the behaviour ledger cannot be read; counts are unknown") from None
        if rows is None:
            replies, operations, coverage = [], [], {"tool_coverage": "unavailable",
                                                     "missing_tool_tables": []}
        else:
            replies, operations, coverage = rows
        coverage.update(ledger_available=rows is not None, owner_configured=bool(instance.owner_id))
        for population in ("reply", "operation"):
            for field in ("platform", "model"):
                coverage[f"{population}_{field}_available"] = bool(coverage.get(f"{population}_{field}_known"))
        return {"since": since, "until": until, "timezone": "UTC", "sources": coverage,
                "replies": reply_counts(replies), "tools": tool_counts(operations),
                "notes": ["Dates include the named UTC days; omitted bounds include all captured history.",
                          "Replies use the last assistant message with non-empty text in each owner turn.",
                          "Canonical work_envelopes in turn-idempotency.db retain A4 capture; the work store "
                          "projection does not retain success or failure_kind.",
                          "Calls and call_failures exclude model_response; request_failures count failed "
                          "model_response operations, with parse_failure counted once per ingress turn.",
                          "Failure rate divides call_failures by calls. Teachable failures include malformed_arguments "
                          "and request parse failures, so they may exceed calls; their rate divides by calls. "
                          "A zero-call denominator has no rate.",
                          "Missing platform or model is unknown; no external database or configured model is used."]}
    finally:
        instance.close()


def behaviour_document(collected: Mapping[str, Any]) -> Document:
    report = Document("Behaviour")
    report.bullets(collected["notes"])
    sources = collected["sources"]
    report.paragraph("Ledger available: " + str(sources["ledger_available"]).lower() +
                     "; owner configured: " + str(sources["owner_configured"]).lower() + ".")
    for population in ("reply", "operation"):
        for field in ("platform", "model"):
            column = sources.get(f"{population}_{field}_column")
            report.paragraph(f"{population.capitalize()} {field}: {column or 'no recorded column'}; "
                             f"available: {str(sources[f'{population}_{field}_available']).lower()}; "
                             f"known rows: {sources.get(f'{population}_{field}_known', 0)}.")
    report.paragraph(f"Skipped rows without a usable time or call identity: {sources.get('skipped_rows', 0)}.")
    report.paragraph(f"Tool coverage: {sources['tool_coverage']}; missing tables: " +
                     (", ".join(sources["missing_tool_tables"]) or "none") + ".")
    report.heading("Replies per UTC day, platform and model")
    for row in collected["replies"]:
        report.paragraph(f"{row['day']} / {row['platform']} / {row['model']}: "
                         f"{row['replies']} replies; {row['replies_with_long_dash']} with long dashes "
                         f"({row['long_dash_reply_share']:.2%}); "
                         f"{row['long_dashes_per_1000_characters']:.3f} long dashes per 1,000 characters.")
        report.paragraph("; ".join(f"{name}: {row[name]}" for name in (
            "characters", "long_dashes", "em_dashes", "en_dashes", "horizontal_bars", "two_em_dashes",
            "three_em_dashes", "small_em_dashes", "en_dash_ranges", "spaced_hyphens", "double_hyphens",
            "replies_with_substitutes")) + f"; substitute reply share: {row['substitute_reply_share']:.2%}.")
    report.heading("Tool capture per UTC day, platform and model")
    for row in collected["tools"]:
        report.paragraph(f"{row['day']} / {row['platform']} / {row['model']}: {row['calls']} calls; "
                         f"{row['call_failures']} call failures; {row['request_failures']} request failures; "
                         f"{row['teachable_failures']} teachable failures; "
                         f"{row['parse_failures']} parse failures; {row['tool_error']} tool_error; "
                         f"{row['unknown_success']} calls without a success value.")
        report.paragraph("Call failures per call: " +
                         (f"{row['failure_rate']:.3f}" if row["failure_rate"] is not None else "unavailable") +
                         "; teachable failures per call: " +
                         (f"{row['teachable_failure_rate']:.3f}" if row["teachable_failure_rate"] is not None
                          else "unavailable") + ".")
        report.paragraph("By kind: " + (", ".join(f"{kind}: {count}" for kind, count in row["by_kind"].items())
                                             or "none") + ".")
        report.paragraph("Top tools by teachable failures: " +
                         (", ".join(f"{tool['tool']}: {tool['teachable_failures']}" for tool in row["top_tools"])
                          or "none") + ".")
    return report


__all__ = ["behaviour_document", "collect", "collect_behaviour", "comparator_line", "document", "faculty",
           "learner_line", "lines", "to_json"]
