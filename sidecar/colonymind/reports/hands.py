"""Read controller counts without creating, migrating or writing its database."""

from pathlib import Path
import sqlite3

from .instance import open_readonly


def collect(config=None, *, path=None):
    if path is None:
        from colonymind.hands.store import DB_FILE
        path = config.home / DB_FILE
    result = {"available": False, "tasks": {}, "hands": {}, "obligations": {}, "evidence": {},
              "unknown_effects": 0, "pending_notices": 0, "events": 0}
    db = None
    try:
        db = open_readonly(Path(path))
        if db is None:
            return result
        db.execute("BEGIN")
        for table in ("tasks", "hands", "obligations"):
            result[table] = {row[0]: row[1] for row in db.execute(f"SELECT state,count(*) FROM {table} GROUP BY state")}
        result["evidence"] = {row[0]: row[1] for row in db.execute(
            "SELECT json_extract(evidence,'$.level'),count(*) FROM hands WHERE evidence IS NOT NULL GROUP BY 1")}
        result["unknown_effects"] = db.execute("SELECT count(*) FROM effects WHERE done_at IS NULL").fetchone()[0]
        result["pending_notices"] = db.execute("SELECT count(*) FROM notices WHERE sent_at IS NULL").fetchone()[0]
        result["events"] = db.execute("SELECT count(*) FROM events").fetchone()[0]
        result["available"] = True
    except (sqlite3.Error, OSError):
        result["error"] = "controller database unreadable"
    finally:
        if db is not None:
            db.close()
    return result


def lines(report):
    if not report["available"]:
        return [report.get("error", "No hands record yet.")]
    output = ["Retained controller totals; delivery receipts mean platform acceptance, not that the owner read a reply."]
    for key in ("tasks", "hands", "obligations", "evidence"):
        output.append(key.capitalize() + ": " + (", ".join(f"{state} {count}" for state, count in sorted(report[key].items())) or "none") + ".")
    output.append(f"Unconfirmed effects: {report['unknown_effects']}; pending notice lines: {report['pending_notices']}.")
    return output


def faculty(reading, section):
    report = collect(reading.instance.config)
    section.lines.extend(lines(report))
    # Retained totals alone do not establish activity in a selected report window.
    section.records = False
