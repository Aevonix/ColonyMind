"""Durable notice batching behind a host-provided, idempotent delivery sink.

The sink receives delivery_id and immutable notice rows. It must persist the
delivery id and return the same platform_message_id on retries. This closes the
crash window between the platform send and this emitter's receipt commit. There
is no default destination or transport, and ordinary replies are never receipts.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable
from urllib.parse import quote
from uuid import uuid4

from .tools import response_body


_sink: Callable | None = None
_lock = threading.RLock()
_emitters: dict[int, "NoticeEmitter"] = {}
_failure_id = "ntc:journal-failure:" + uuid4().hex
_failure_notice = None
_failure_receipt = None
_failure_attempts: dict[str, int] = {}


def register_notice_sink(sink: Callable | None) -> None:
    """Install the host's delivery seam; None explicitly disables delivery."""
    if sink is not None and not callable(sink):
        raise TypeError("Notice sink must be callable")
    global _sink
    with _lock:
        _sink = sink
        _emitters.clear()


def journal_failure(runtime=None) -> dict:
    """One actionable sink notice when the normal durable journal is unusable.

    No journal or controller is required. The host sink owns durable deduplication
    of this immutable delivery id. The process latches an actual receipt; absent
    a journal it can only bound uncertain retries to twenty per local day.
    """
    global _failure_notice, _failure_receipt
    try:
        with _lock:
            if _failure_receipt:
                return {"platform_message_id": _failure_receipt}
            sink = _sink or getattr(runtime, "notice_sink", None)
            if not callable(sink):
                return {"disabled": True}
            current = datetime.now().astimezone()
            day = current.date().isoformat()
            if _failure_attempts.get(day, 0) >= 20:
                return {"limited": True}
            if _failure_notice is None:
                _failure_notice = {"id": _failure_id, "budget": "actionable",
                                   "line": "Hands journal unavailable; new work is refused.",
                                   "queued_at": current.isoformat(), "hand_id": "", "attempt": 0}
            _failure_attempts[day] = _failure_attempts.get(day, 0) + 1
            result = sink(delivery_id=_failure_id, notices=[dict(_failure_notice)])
            receipt = result.get("platform_message_id") if isinstance(result, dict) else None
            if isinstance(receipt, str) and receipt.strip():
                _failure_receipt = receipt
                return {"platform_message_id": receipt}
            return {"delivery": "unknown"}
    except Exception:
        return {"delivery": "unknown"}


def _instant(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def compose(notices: list[dict], *, forbidden_patterns=()) -> str:
    """Validate transport bounds and the host's optional text validators."""
    if (not isinstance(forbidden_patterns, (list, tuple))
            or any(not isinstance(pattern, str) or not pattern for pattern in forbidden_patterns)):
        raise ValueError("Notice validators must be a list of nonempty regular expressions")
    validators = [re.compile(pattern, re.IGNORECASE) for pattern in forbidden_patterns]
    lines = []
    for notice in notices:
        line = notice.get("line")
        if (not isinstance(line, str) or not line.strip() or len(line) > 200
                or "\r" in line or "\n" in line or any(pattern.search(line) for pattern in validators)):
            raise ValueError("Notice line exceeds transport bounds or fails a host validator")
        lines.append(line)
    return "\n".join(lines)


# hands.notices defaults: the daily caps are the owner's D3 choice; the merge and overflow timings are assumptions.
NOTICE_DEFAULTS = {"routine_per_day": 30, "actionable_per_day": 20, "merge_seconds": 60, "overflow_seconds": 3600}


class NoticeEmitter:
    def __init__(self, *, client, emit, sink, ledger_path, now=time.time, settings=None):
        self.client, self.emit, self.sink, self.now = client, emit, sink, now
        self._settings = settings or (lambda: {})
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS pending (
                    id TEXT PRIMARY KEY, budget TEXT NOT NULL, payload TEXT NOT NULL,
                    queued REAL NOT NULL, batch TEXT, acknowledged INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, budget TEXT NOT NULL, day TEXT NOT NULL,
                    payload TEXT NOT NULL, receipt TEXT, summary INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS budget_days (
                    batch TEXT NOT NULL, day TEXT NOT NULL, budget TEXT NOT NULL,
                    summary INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(batch,day)
                );
                INSERT OR IGNORE INTO budget_days(batch,day,budget,summary)
                    SELECT id,day,budget,summary FROM batches;
            """)
            db.commit()

    def limits(self) -> dict:
        values = dict(NOTICE_DEFAULTS)
        try:
            values.update({key: value for key, value in (self._settings() or {}).items()
                           if key in NOTICE_DEFAULTS and isinstance(value, (int, float)) and value >= 0})
        except Exception:
            pass
        return values

    def _maximum(self, budget: str) -> int:
        limits = self.limits()
        return int(limits["routine_per_day"] if budget == "routine" else limits["actionable_per_day"])

    def _compose(self, rows):
        return compose(rows, forbidden_patterns=(self._settings() or {}).get("forbidden_patterns", []))

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _ingest(self, db):
        try:
            result = response_body(self.client.get("/v1/hands/notices", params={"cursor": ""}))
        except Exception:
            return
        for row in result.get("notices", []):
            try:
                self._compose([row])
                if (row.get("budget") not in {"routine", "actionable"} or not row.get("id")
                        or not row.get("hand_id") or not isinstance(row.get("attempt"), int)):
                    continue
                # A controller row already carrying an actual platform id needs no send.
                if row.get("platform_message_id"):
                    continue
                db.execute("INSERT OR IGNORE INTO pending(id,budget,payload,queued) VALUES(?,?,?,?)",
                           (row["id"], row["budget"], json.dumps(row, sort_keys=True), _instant(row["queued_at"])))
            except (ValueError, TypeError, KeyError):
                continue
        db.commit()

    def _acknowledge(self, db):
        for row in db.execute("""SELECT pending.id, batches.receipt FROM pending
                JOIN batches ON pending.batch=batches.id
                WHERE pending.acknowledged=0 AND batches.receipt IS NOT NULL""").fetchall():
            try:
                # emit journals this write before POST, with the same key on retry.
                self.emit(f"/v1/hands/notices/{quote(row['id'], safe='')}/sent",
                          {"platform_message_id": row["receipt"]}, f"notice-sent:{row['id']}")
                db.execute("UPDATE pending SET acknowledged=1 WHERE id=?", (row["id"],))
                db.commit()
            except Exception:
                continue

    def _deliver(self, db, batch):
        try:
            day = datetime.fromtimestamp(self.now()).date().isoformat()
            reserved = db.execute("SELECT 1 FROM budget_days WHERE batch=? AND day=?",
                                  (batch["id"], day)).fetchone()
            if reserved is None:
                # A retry after midnight might make its first successful send
                # today. Reserve both dates conservatively rather than allowing
                # old uncertain batches to bypass today's daily limit.
                count = db.execute("SELECT count(*) FROM budget_days WHERE budget=? AND day=? AND summary=0",
                                   (batch["budget"], day)).fetchone()[0]
                if not batch["summary"] and count >= self._maximum(batch["budget"]):
                    return False
                db.execute("INSERT INTO budget_days(batch,day,budget,summary) VALUES(?,?,?,?)",
                           (batch["id"], day, batch["budget"], batch["summary"]))
                db.commit()
            rows = json.loads(batch["payload"])
            self._compose(rows)
            receipt = self.sink(delivery_id=batch["id"], notices=rows)
            message_id = receipt.get("platform_message_id") if isinstance(receipt, dict) else None
            if not isinstance(message_id, str) or not message_id.strip():
                return False
            db.execute("UPDATE batches SET receipt=? WHERE id=?", (message_id, batch["id"]))
            db.commit()
            return True
        except Exception:
            # Delivery may already have happened. Keep the exact batch id for the
            # idempotent host sink; never invent a receipt or a replacement batch.
            return False

    def _batch(self, db, rows, budget, day, now, *, summary=False):
        payload = [json.loads(row["payload"]) for row in rows]
        if summary:
            hour = int(now // 3600)
            payload = [{"id": f"ntc:overflow:{day}:{hour}", "budget": "actionable",
                        "line": f"{len(rows)} more events; /hands", "hand_id": "", "attempt": 0,
                        "notice_ids": [item["id"] for item in payload]}]
        identity = json.dumps([budget, day, payload], sort_keys=True)
        batch_id = "ntc-batch:" + hashlib.sha256(identity.encode()).hexdigest()
        db.execute("INSERT OR IGNORE INTO batches(id,budget,day,payload,summary,created) VALUES(?,?,?,?,?,?)",
                   (batch_id, budget, day, json.dumps(payload, sort_keys=True), int(summary), now))
        db.execute("INSERT OR IGNORE INTO budget_days(batch,day,budget,summary) VALUES(?,?,?,?)",
                   (batch_id, day, budget, int(summary)))
        if not summary:
            db.executemany("UPDATE pending SET batch=? WHERE id=?", [(batch_id, row["id"]) for row in rows])
        # Reserve the budget and id before sending. An uncertain sink result
        # consumes its reservation until that same sink id supplies a receipt.
        db.commit()
        return db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()

    def pump(self) -> dict:
        """Pull pending rows and send mature batches; all host failures are contained."""
        try:
            with _lock, closing(self._connect()) as db:
                now = self.now()
                day = datetime.fromtimestamp(now).date().isoformat()
                self._ingest(db)
                self._acknowledge(db)
                uncertain = db.execute("SELECT * FROM batches WHERE receipt IS NULL ORDER BY created").fetchall()
                # An unresolved send stops that budget stream. Repeated failures
                # cannot allocate new delivery ids and multiply uncertain sends.
                blocked = set()
                for batch in uncertain:
                    if not self._deliver(db, batch):
                        blocked.add(batch["budget"])
                limits = self.limits()
                for budget in ("routine", "actionable"):
                    maximum = self._maximum(budget)
                    if budget in blocked:
                        continue
                    rows = db.execute("SELECT * FROM pending WHERE batch IS NULL AND budget=? ORDER BY queued,id",
                                      (budget,)).fetchall()
                    if not rows or now - rows[0]["queued"] < limits["merge_seconds"]:
                        continue
                    count = db.execute("SELECT count(*) FROM budget_days WHERE budget=? AND day=? AND summary=0",
                                       (budget, day)).fetchone()[0]
                    if count >= maximum:
                        if budget == "actionable":
                            previous = db.execute("SELECT max(created) FROM batches WHERE summary=1").fetchone()[0]
                            if previous is None or now - previous >= limits["overflow_seconds"]:
                                self._deliver(db, self._batch(db, rows, budget, day, now, summary=True))
                        continue
                    self._deliver(db, self._batch(db, rows, budget, day, now))
                self._acknowledge(db)
            return {"ok": True}
        except Exception:
            return {"error": "Notice delivery unavailable"}


def pump(runtime) -> dict:
    """Called by executor maintenance; configuration alone cannot choose a recipient."""
    with _lock:
        sink = _sink or getattr(runtime, "notice_sink", None)
        if not callable(sink):
            return {"disabled": True}
        emitter = _emitters.get(id(runtime))
        if emitter is None:
            try:
                emitter = NoticeEmitter(client=runtime.client, emit=runtime.emit, sink=sink,
                                        ledger_path=runtime.spool_dir / "notice-delivery.sqlite3",
                                        settings=lambda: (getattr(runtime, "config", {}) or {}).get("notices", {}))
            except Exception:
                return {"error": "Notice ledger unavailable"}
            _emitters[id(runtime)] = emitter
        return emitter.pump()


def register(ctx: Any) -> None:
    """The runtime's maintenance loop pumps notices only after a sink is registered."""
    return None
