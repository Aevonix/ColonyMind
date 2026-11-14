"""Export standing evidence and tool names, never content, credentials or real ids.

Run with Python -B on the host and redirect stdout to a private local fixture.
Every SQLite store is opened mode=ro. Identity joins use salted hashes held only
in memory; exported references are freshly assigned synthetic labels. No host
code is imported, no endpoint is called, and no store is created or migrated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.parse
import urllib.request

import yaml


def read_store(path, sql, params=()):
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        return [dict(row) for row in db.execute(sql, params)]


def object_of(value):
    try:
        return json.loads(value or "{}")
    except (ValueError, TypeError):
        return {}


def export(instance: Path | None, hermes: Path, days: int, *, read_cron_standing: bool = False):
    salt = os.urandom(32)
    labels = {}

    def ref(value):
        if not value:
            return ""
        digest = hashlib.sha256(salt + str(value).encode()).digest()
        return labels.setdefault(digest, "session-" + str(len(labels) + 1))

    config = yaml.safe_load((hermes / "config.yaml").read_text()) or {}
    section = config.get("plugins", {}).get("colonymind", {})
    if instance is None:
        instance = Path(section.get("key_file") or Path.home() / ".colonymind/api.key").expanduser().parent
    identity_paths = [instance / "identity.yaml", instance / "data/identity.yaml"]
    identity = next((yaml.safe_load(p.read_text()) or {} for p in identity_paths if p.is_file()), {})
    instance_config = yaml.safe_load((instance / "colonymind.yaml").read_text()) or {}
    owner_contact = (instance_config.get("owner", {}).get("contact_id")
                     or identity.get("owner", {}).get("contact_id") or section.get("owner_contact_id"))
    raw_handles = identity.get("owner", {}).get("handles", [])
    owner_handles = set()
    if isinstance(raw_handles, dict):
        raw_handles = [{"platform": k, "id": v} for k, values in raw_handles.items()
                       for v in (values if isinstance(values, list) else [values])]
    for entry in raw_handles:
        if isinstance(entry, str):
            platform, _, sender = entry.replace(":", "=", 1).partition("=")
        else:
            platform, sender = entry.get("platform", ""), entry.get("id") or entry.get("handle") or ""
        owner_handles.add((str(platform).lower(), str(sender).lower()))
    for platform in ("sms", "rcs"):
        value = instance_config.get("owner", {}).get(platform + "_handle")
        if isinstance(value, str) and re.fullmatch(r"\+[1-9][0-9]{6,14}", value.strip()):
            owner_handles.add((platform, value.strip().lower()))
    data = instance / "data" if (instance / "data/hands.db").exists() else instance
    if owner_contact:
        for handle in read_store(data / "contacts.db", "SELECT gateway,address FROM contact_handles WHERE contact_id=?",
                                 (owner_contact,)):
            owner_handles.add((str(handle["gateway"]).lower(), str(handle["address"]).lower()))

    def sender_standing(platform, sender):
        if not sender:
            return "unknown"
        return "owner" if (str(platform).lower(), str(sender).lower()) in owner_handles else "guest"

    # These are structural stores, not a replay model's guesses about names.
    hand_rows = read_store(data / "hands.db", "SELECT session_id,parent_session,hand_id FROM hand_sessions")
    hand_rows += read_store(data / "hands.db", "SELECT session_id,parent_session,hand_id FROM hand_session_tombstones")
    hands = read_store(data / "hands.db", "SELECT id,task_id,parent_session,child_session FROM hands")
    tasks = read_store(data / "hands.db", "SELECT id,origin,origin_ref FROM tasks")
    board = read_store(hermes / "kanban.db", "SELECT id,session_id,idempotency_key,created_by FROM tasks")
    initiatives = read_store(data / "initiatives.db", "SELECT id,hermes_ref,created_by FROM initiatives")
    mind_rows = read_store(data / "mind.db", "SELECT key,text FROM mind_state WHERE key LIKE 'cron_%'")
    cron_data = object_of((hermes / "cron/jobs.json").read_text()) if (hermes / "cron/jobs.json").exists() else {}
    jobs = cron_data.get("jobs", []) if isinstance(cron_data, dict) else cron_data
    sessions = read_store(hermes / "state.db", "SELECT id,source,user_id,parent_session_id,origin_json,model_config FROM sessions")
    by_id = {row["id"]: row for row in sessions}
    hand_parents = {row["session_id"]: row["parent_session"] for row in hand_rows}
    hand_parents.update({row["child_session"]: row["parent_session"] for row in hands if row["child_session"]})
    jobs_by_id = {str(job["id"]): job for job in jobs}
    cron_standing = {}
    if read_cron_standing:
        # The operator must verify that this read-only standing route exists
        # before enabling this option. No other endpoint is contacted.
        url = section.get("sidecar_url") or section.get("url")
        credential = Path(section["key_file"]).expanduser().read_text().strip().splitlines()[0]
        for job in jobs:
            query = urllib.parse.urlencode({"job_id": job["id"], "created_at": job.get("created_at", "")})
            request = urllib.request.Request(url.rstrip("/") + "/v1/mind/cron-origin?" + query,
                                             headers={"Authorization": "Bearer " + credential}, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    origin = json.load(response).get("origin")
                # The old owner classification is only absence of a mind mark.
                if origin == "mind":
                    cron_standing[str(job["id"])] = origin
            except Exception:
                pass
        del credential
    # The retained memory admission ledger is evidence that the host accepted
    # this exact native session for this contact, not a platform-name guess.
    admitted_owner = set()
    ownership_count = 0
    for path in (hermes / "state").glob("*.sqlite3"):
        tables = read_store(path, "SELECT name FROM sqlite_master WHERE type='table' AND name='native_source_ownership'")
        if tables and owner_contact:
            records = read_store(path, "SELECT session_id FROM native_source_ownership WHERE contact_id=?",
                                 (owner_contact,))
            ownership_count += len(records)
            admitted_owner.update(row["session_id"] for row in records)
    # Keep sidecar cron provenance, including unconfirmed changes, in memory.
    cron_marks = {row["key"]: object_of(row["text"]) for row in mind_rows}
    board_standing = {row["session_id"]: ("mind" if str(row["idempotency_key"] or "").startswith("mind:")
                                         else "unknown") for row in board if row["session_id"]}
    selected = {"colonymind_task", "subagent", "cron", "companion"}
    resolved = {}

    def resolve(sid, seen=()):
        if sid in resolved:
            return resolved[sid]
        if not sid or sid in seen or len(seen) >= 32:
            return "unknown", "parent_missing_or_cyclic"
        row = by_id.get(sid)
        parent = hand_parents.get(sid) or (row or {}).get("parent_session_id")
        if parent:
            run, reason = resolve(parent, (*seen, sid))
            if run != "unknown":
                resolved[sid] = run, "retained_hand_parent" if sid in hand_parents else "session_parent"
                return resolved[sid]
        if row is None:
            return "unknown", "parent_not_retained"
        source = row["source"]
        origin = object_of(row["origin_json"])
        platform = origin.get("platform") or source
        sender = origin.get("user_id") or row["user_id"]
        run = sender_standing(platform, sender)
        if source in {"colonymind_task", "companion"} and sid in admitted_owner and (
                sender == owner_contact or any(str(sender).lower() == value for _, value in owner_handles)):
            resolved[sid] = "owner", "retained_owner_admission"
        elif run == "owner":
            resolved[sid] = run, "recorded_sender"
        elif source == "cli" and not sender:
            resolved[sid] = "owner", "terminal_root"
        elif source == "cron":
            match = re.match(r"^cron_(.+)_\d{8}_\d{6}$", sid)
            job = jobs_by_id.get(match.group(1)) if match else None
            if job:
                # Delivery origin is not creator authority. Only a recorded
                # creator session or sidecar mind provenance establishes it.
                parent = job.get("context_from")
                if str(job["id"]) in cron_standing:
                    resolved[sid] = cron_standing[str(job["id"])], "live_cron_standing"
                elif parent and isinstance(parent, str) and parent in by_id:
                    resolved[sid] = resolve(parent, (*seen, sid))
                elif any(str(job["id"]) in key and "job" in key for key in cron_marks):
                    resolved[sid] = "mind", "sidecar_cron_origin"
                else:
                    resolved[sid] = "unknown", "cron_creator_not_recorded"
            else:
                resolved[sid] = "unknown", "cron_job_not_retained"
        elif board_standing.get(sid) == "mind":
            resolved[sid] = "mind", "sidecar_kanban_origin"
        elif source in selected:
            resolved[sid] = "unknown", "automation_parent_unresolved"
        else:
            resolved[sid] = run, "recorded_sender" if sender else "sender_not_recorded"
        return resolved[sid]

    cutoff = time.time() - days * 86400
    # JSON1 projects function names only; no message content or tool arguments
    # leave SQLite. The timestamp filter covers calls in older retained sessions.
    calls = read_store(hermes / "state.db", """
        SELECT m.session_id, json_extract(j.value,'$.function.name') AS tool
        FROM messages m, json_each(m.tool_calls) j
        WHERE m.timestamp>=? AND m.role='assistant' AND json_valid(m.tool_calls)
        ORDER BY m.timestamp,m.id,j.key
    """, (cutoff,))
    output = []
    fixture_sessions = {}
    for call in calls:
        sid = call["session_id"]
        row = by_id.get(sid)
        if not row or row["source"] not in selected or not call["tool"]:
            continue
        run, reason = resolve(sid)
        label = ref(sid)
        fixture_sessions[label] = {"class": row["source"], "standing": run, "reason": reason}
        output.append({"session": label, "tool": call["tool"]})
    return {"days": days, "stores": {"hand_sessions": len(hand_rows), "hands": len(hands), "tasks": len(tasks),
            "kanban": len(board), "initiatives": len(initiatives), "cron": len(jobs),
            "owner_admissions": ownership_count, "live_cron_standings": len(cron_standing)},
            "sessions": fixture_sessions, "calls": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-home", type=Path)
    parser.add_argument("--hermes-home", type=Path, default=Path.home() / ".hermes")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--read-cron-standing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.instance_home.resolve() if args.instance_home else None,
                            args.hermes_home.resolve(), args.days, read_cron_standing=args.read_cron_standing)))
