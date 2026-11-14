"""CompetenceStore -- per-domain outcome counts + latency, from real work.

Domains are capability classes: initiative types ("research", "follow_up"),
"project" (project steps/outcomes), "directed" (directed-action audits),
"delivery" (real pushes), and worker job types (item 5). Every recording site
is a completion/failure chokepoint that already exists; the store never polls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Exponential moving average weight for latency (recent work dominates).
_EWMA_ALPHA = 0.3

_OUTCOMES = ("success", "failure", "timeout")


def _norm_outcome(outcome: str) -> str:
    o = (outcome or "").strip().lower()
    if o in ("success", "completed", "clean", "actioned", "done", "ok"):
        return "success"
    if o in ("timeout", "timed_out"):
        return "timeout"
    return "failure"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def event_fingerprint(event: Dict[str, Any]) -> str:
    """Stable identity for one immutable raw competence event.

    The reconciliation tool requires this fingerprint as well as the row id,
    so a plan made against one database cannot silently target a different
    event after a restore or migration.
    """
    evidence = event.get("evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except ValueError:
            pass
    payload = {
        "id": int(event["id"]),
        "domain": str(event["domain"]),
        "outcome": str(event["outcome"]),
        "shadow": int(event.get("shadow") or 0),
        "violation": int(event.get("violation") or 0),
        "stated_confidence": event.get("stated_confidence"),
        "source": event.get("source") or "legacy_unattributed",
        "source_ref": event.get("source_ref"),
        "evidence_status": (
            event.get("evidence_status") or "legacy_unattributed"),
        "outcome_contract": (
            event.get("outcome_contract") or "legacy.unversioned"),
        "evidence": evidence,
        "ts": float(event["ts"]),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class CompetenceStore:
    """SQLite persistence of per-domain success/failure/timeout + ewma latency."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path) if db_path else ":memory:",
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS competence (
                    domain TEXT PRIMARY KEY,
                    success INTEGER DEFAULT 0,
                    failure INTEGER DEFAULT 0,
                    timeout INTEGER DEFAULT 0,
                    ewma_latency_secs REAL,
                    last_outcome TEXT,
                    last_outcome_at REAL
                )""")
            # Per-event log for windowed circuit-breaker queries and
            # calibration-vs-real evidence separation (shadow=1 events are
            # calibration runs; they graduate a class out of shadow but never
            # count toward act-first confidence).
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS competence_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL, outcome TEXT NOT NULL,
                    shadow INTEGER DEFAULT 0, violation INTEGER DEFAULT 0,
                    stated_confidence REAL,
                    source TEXT,
                    source_ref TEXT,
                    event_key TEXT,
                    evidence_status TEXT,
                    outcome_contract TEXT,
                    evidence TEXT,
                    ts REAL NOT NULL
                )""")
            # Additive migrations: old events stay byte-for-byte intact and
            # are explicitly labelled legacy_unattributed on the read side.
            try:
                cols = {r[1] for r in self._conn.execute(
                    "PRAGMA table_info(competence_events)").fetchall()}
                additions = {
                    "stated_confidence": "REAL",
                    "source": "TEXT",
                    "source_ref": "TEXT",
                    "event_key": "TEXT",
                    "evidence_status": "TEXT",
                    "outcome_contract": "TEXT",
                    "evidence": "TEXT",
                }
                for name, sql_type in additions.items():
                    if name in cols:
                        continue
                    self._conn.execute(
                        "ALTER TABLE competence_events "
                        f"ADD COLUMN {name} {sql_type}")
            except Exception:
                pass
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cev_domain_ts "
                "ON competence_events(domain, ts)")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_cev_source_event_key "
                "ON competence_events(source, event_key) "
                "WHERE event_key IS NOT NULL")
            # This is the sole correction mechanism. Rows are append-only;
            # the raw event and the all-time aggregate are never rewritten.
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS competence_reconciliations (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    reconciliation_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    event_id INTEGER,
                    target_fingerprint TEXT,
                    disposition TEXT,
                    replacement_outcome TEXT,
                    domain TEXT,
                    since_ts REAL,
                    until_ts REAL,
                    supersedes TEXT,
                    reason TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    CHECK (kind IN ('event_correction', 'evidence_gap',
                                    'gap_resolution')),
                    CHECK (disposition IS NULL OR
                           disposition IN ('invalidate', 'replace'))
                )""")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_crec_supersedes "
                "ON competence_reconciliations(supersedes) "
                "WHERE supersedes IS NOT NULL")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_crec_event "
                "ON competence_reconciliations(event_id, seq)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_crec_gap "
                "ON competence_reconciliations(domain, since_ts, until_ts)")
            self._conn.execute(
                """CREATE TRIGGER IF NOT EXISTS trg_crec_no_update
                   BEFORE UPDATE ON competence_reconciliations
                   BEGIN
                     SELECT RAISE(ABORT, 'competence ledger is append-only');
                   END""")
            self._conn.execute(
                """CREATE TRIGGER IF NOT EXISTS trg_crec_no_delete
                   BEFORE DELETE ON competence_reconciliations
                   BEGIN
                     SELECT RAISE(ABORT, 'competence ledger is append-only');
                   END""")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(self, domain: str, outcome: str,
               latency_secs: Optional[float] = None,
               shadow: bool = False, violation: bool = False,
               stated_confidence: Optional[float] = None, *,
               source: Optional[str] = None,
               source_ref: Optional[str] = None,
               event_key: Optional[str] = None,
               evidence_status: str = "observed",
               outcome_contract: str = "legacy.unversioned",
               evidence: Optional[Dict[str, Any]] = None) -> bool:
        """Record one outcome for a domain. Never raises.

        stated_confidence: the confidence the model STATED before doing the
        work (charter contract); stored per event so calibration (stated vs
        realized) is measurable and autonomy is earned against it.
        """
        try:
            domain = str(domain or "unknown").strip().lower()
            outcome = _norm_outcome(str(outcome or ""))
            source = str(source or "legacy_callsite").strip()[:128]
            source_ref = str(source_ref or "").strip()[:512] or None
            event_key = str(event_key or "").strip()[:512] or None
            evidence_status = str(
                evidence_status or "observed").strip().lower()
            if evidence_status not in {
                    "verified", "observed", "unverified",
                    "legacy_unattributed"}:
                evidence_status = "unverified"
            outcome_contract = str(
                outcome_contract or "legacy.unversioned").strip()[:128]
            evidence_json = _canonical_json(evidence) if evidence else None
        except Exception as exc:
            logger.debug("competence record normalization failed: %s", exc)
            return False
        now = time.time()
        try:
            with self._lock:
                if event_key is not None:
                    duplicate = self._conn.execute(
                        """SELECT 1 FROM competence_events
                           WHERE source = ? AND event_key = ? LIMIT 1""",
                        (source, event_key),
                    ).fetchone()
                    if duplicate is not None:
                        return False
                row = self._conn.execute(
                    "SELECT ewma_latency_secs FROM competence WHERE domain=?",
                    (domain,)).fetchone()
                ewma = row["ewma_latency_secs"] if row else None
                if latency_secs is not None:
                    ewma = (float(latency_secs) if ewma is None else
                            _EWMA_ALPHA * float(latency_secs) + (1 - _EWMA_ALPHA) * float(ewma))
                self._conn.execute(
                    f"""INSERT INTO competence
                        (domain, {outcome}, ewma_latency_secs, last_outcome, last_outcome_at)
                        VALUES (?, 1, ?, ?, ?)
                        ON CONFLICT(domain) DO UPDATE SET
                          {outcome}={outcome}+1, ewma_latency_secs=?,
                          last_outcome=?, last_outcome_at=?""",
                    (domain, ewma, outcome, now, ewma, outcome, now))
                self._conn.execute(
                    "INSERT INTO competence_events (domain, outcome, shadow, "
                    "violation, stated_confidence, source, source_ref, event_key, "
                    "evidence_status, outcome_contract, evidence, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (domain, outcome, 1 if shadow else 0,
                     1 if violation else 0, stated_confidence, source,
                     source_ref, event_key, evidence_status, outcome_contract,
                     evidence_json, now))
                # Do not prune evidence here. The reconciliation ledger must
                # remain auditable against the immutable event it targets.
                self._conn.commit()
            return True
        except Exception as exc:
            # The aggregate update and immutable event insert are one logical
            # evidence write. Never leave an uncommitted aggregate increment
            # that a later, unrelated record could accidentally commit.
            try:
                with self._lock:
                    self._conn.rollback()
            except Exception:
                logger.debug(
                    "competence rollback failed for %s", domain,
                    exc_info=True,
                )
            logger.debug("competence record failed for %s: %s", domain, exc)
            return False

    # -- append-only reconciliation -------------------------------------
    @staticmethod
    def _decode_json(value: Any) -> Any:
        if value in (None, ""):
            return None
        try:
            return json.loads(str(value))
        except (TypeError, ValueError):
            return value

    def _ledger_rows(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM competence_reconciliations ORDER BY seq ASC"
        ).fetchall()
        out = []
        for raw in rows:
            row = dict(raw)
            row["provenance"] = self._decode_json(row.get("provenance"))
            out.append(row)
        return out

    @staticmethod
    def _effective_corrections(
            rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        effective: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            if row.get("kind") != "event_correction":
                continue
            event_id = row.get("event_id")
            if event_id is not None:
                effective[int(event_id)] = row
        return effective

    def active_evidence_gaps(
            self, domain: Optional[str] = None,
            since: Optional[float] = None,
            until: Optional[float] = None) -> List[Dict[str, Any]]:
        """Unresolved half-open windows whose events cannot be trusted.

        A gap is deliberately broader than an event invalidation: it says the
        available provenance cannot identify the contaminated rows exactly.
        Consumers must report the affected metric unavailable, not guess.
        """
        wanted = (domain or "").strip().lower() or None
        with self._lock:
            ledger = self._ledger_rows()
        resolved = {
            str(r.get("supersedes")) for r in ledger
            if r.get("kind") == "gap_resolution" and r.get("supersedes")
        }
        out = []
        for row in ledger:
            if row.get("kind") != "evidence_gap":
                continue
            if row["reconciliation_id"] in resolved:
                continue
            if wanted is not None and row.get("domain") != wanted:
                continue
            gap_since = float(row["since_ts"])
            gap_until = float(row["until_ts"])
            if since is not None and gap_until <= float(since):
                continue
            if until is not None and gap_since >= float(until):
                continue
            out.append(row)
        return out

    def reconciliation_revision(
            self, domains: Optional[Iterable[str]] = None,
            since: Optional[float] = None,
            until: Optional[float] = None) -> int:
        """Highest ledger sequence affecting the requested evidence slice."""
        wanted = None
        if domains is not None:
            wanted = {(d or "").strip().lower() for d in domains if d}
        with self._lock:
            ledger = self._ledger_rows()
            events = {int(r["id"]): dict(r) for r in self._conn.execute(
                "SELECT id, domain, ts FROM competence_events").fetchall()}
        gaps = {r["reconciliation_id"]: r for r in ledger
                if r.get("kind") == "evidence_gap"}

        def in_slice(domain: str, start: float, end: float) -> bool:
            if wanted is not None and domain not in wanted:
                return False
            if since is not None and end <= float(since):
                return False
            if until is not None and start >= float(until):
                return False
            return True

        def point_in_slice(domain: str, ts: float) -> bool:
            if wanted is not None and domain not in wanted:
                return False
            if since is not None and ts < float(since):
                return False
            if until is not None and ts >= float(until):
                return False
            return True

        revision = 0
        for row in ledger:
            kind = row.get("kind")
            if kind == "event_correction":
                ev = events.get(int(row.get("event_id") or -1))
                if ev and point_in_slice(
                        str(ev["domain"]), float(ev["ts"])):
                    revision = max(revision, int(row["seq"]))
            elif kind == "evidence_gap":
                if in_slice(str(row["domain"]), float(row["since_ts"]),
                            float(row["until_ts"])):
                    revision = max(revision, int(row["seq"]))
            elif kind == "gap_resolution":
                gap = gaps.get(str(row.get("supersedes") or ""))
                if gap and in_slice(str(gap["domain"]),
                                    float(gap["since_ts"]),
                                    float(gap["until_ts"])):
                    revision = max(revision, int(row["seq"]))
        return revision

    def events(self, domain: str, since: Optional[float] = None,
               include_shadow: bool = True, *,
               until: Optional[float] = None,
               include_invalid: bool = False,
               include_unavailable: bool = False,
               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Effective events for a domain, newest first.

        Exact invalidations/replacements and unresolved evidence gaps are
        applied at read time. Raw rows are never updated. There is no default
        limit: weekly scorecards must not silently truncate a busy domain at
        the old 1,000-event ceiling.
        """
        domain = (domain or "").strip().lower()
        q = "SELECT * FROM competence_events WHERE domain=?"
        params: List[Any] = [domain]
        if since is not None:
            q += " AND ts >= ?"
            params.append(float(since))
        if until is not None:
            q += " AND ts < ?"
            params.append(float(until))
        if not include_shadow:
            q += " AND shadow=0"
        q += " ORDER BY ts DESC, id DESC"
        if limit is not None:
            q += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            raw_rows = [dict(r) for r in self._conn.execute(q, params).fetchall()]
            ledger = self._ledger_rows()
        corrections = self._effective_corrections(ledger)
        gaps = self.active_evidence_gaps(domain, since=since, until=until)
        out: List[Dict[str, Any]] = []
        for row in raw_rows:
            row["source"] = row.get("source") or "legacy_unattributed"
            row["evidence_status"] = (
                row.get("evidence_status") or "legacy_unattributed")
            row["outcome_contract"] = (
                row.get("outcome_contract") or "legacy.unversioned")
            row["evidence"] = self._decode_json(row.get("evidence"))
            row["fingerprint"] = event_fingerprint(row)
            row["valid"] = True
            row["evidence_available"] = True
            correction = corrections.get(int(row["id"]))
            if correction is not None:
                row["recorded_outcome"] = row["outcome"]
                row["reconciliation"] = {
                    "id": correction["reconciliation_id"],
                    "disposition": correction["disposition"],
                    "reason": correction["reason"],
                    "provenance": correction["provenance"],
                    "created_by": correction["created_by"],
                    "created_at": correction["created_at"],
                }
                if correction["disposition"] == "invalidate":
                    row["valid"] = False
                else:
                    row["outcome"] = correction["replacement_outcome"]
            covering = [g for g in gaps
                        if float(g["since_ts"]) <= float(row["ts"])
                        < float(g["until_ts"])]
            if covering:
                row["evidence_available"] = False
                row["evidence_gap_ids"] = [
                    g["reconciliation_id"] for g in covering]
            if not row["valid"] and not include_invalid:
                continue
            if not row["evidence_available"] and not include_unavailable:
                continue
            out.append(row)
        return out


    def calibration(self, domain: str) -> Optional[Dict[str, Any]]:
        """Stated-vs-realized calibration for a domain: mean absolute error
        between the confidence the model stated and the realized outcome
        (success=1, else 0), over non-shadow events that stated one."""
        events = [e for e in self.events(domain, include_shadow=False)
                  if e.get("stated_confidence") is not None]
        if not events:
            return None
        errs = [abs(float(e["stated_confidence"])
                    - (1.0 if e["outcome"] == "success" else 0.0))
                for e in events]
        realized = sum(1.0 for e in events
                       if e["outcome"] == "success") / len(events)
        return {"n": len(events),
                "mean_abs_error": round(sum(errs) / len(errs), 3),
                "mean_stated": round(sum(float(e["stated_confidence"])
                                         for e in events) / len(events), 3),
                "mean_realized": round(realized, 3)}

    def get(self, domain: str) -> Optional[Dict[str, Any]]:
        domain = (domain or "").strip().lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM competence WHERE domain=?", (domain,)).fetchone()
        return self._annotate_domain(dict(row)) if row else None

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM competence ORDER BY last_outcome_at DESC").fetchall()
        return [self._annotate_domain(dict(r)) for r in rows]

    def _annotate_domain(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ledger deltas without rewriting the legacy aggregate row."""
        domain = str(d.get("domain") or "")
        base_n = sum(int(d.get(o) or 0) for o in _OUTCOMES)
        effective = self.events(
            domain, include_shadow=True, include_invalid=True,
            include_unavailable=True)
        excluded_ids = set()
        corrected = 0
        for event in effective:
            reconciliation = event.get("reconciliation")
            if reconciliation is not None:
                corrected += 1
                recorded = str(event.get("recorded_outcome") or "")
                if recorded in _OUTCOMES:
                    d[recorded] = max(0, int(d.get(recorded) or 0) - 1)
                if event.get("valid"):
                    replacement = str(event.get("outcome") or "")
                    if replacement in _OUTCOMES:
                        d[replacement] = int(d.get(replacement) or 0) + 1
                else:
                    excluded_ids.add(int(event["id"]))
            if (event.get("valid")
                    and not event.get("evidence_available", True)):
                outcome = str(event.get("outcome") or "")
                if outcome in _OUTCOMES:
                    d[outcome] = max(0, int(d.get(outcome) or 0) - 1)
                excluded_ids.add(int(event["id"]))
        gaps = self.active_evidence_gaps(domain)
        available = self.events(domain, include_shadow=True)
        if available:
            d["last_outcome"] = available[0]["outcome"]
            d["last_outcome_at"] = available[0]["ts"]
        elif base_n:
            d["last_outcome"] = None
            d["last_outcome_at"] = None
        n = sum(int(d.get(o) or 0) for o in _OUTCOMES)
        d["n"] = n
        # A declared gap means even the denominator is not knowable. Expose
        # the remaining evidence counts, but never turn them into a rate.
        d["success_rate"] = (
            round(int(d.get("success") or 0) / n, 3)
            if n and not gaps else None)
        d["timeout_rate"] = (
            round(int(d.get("timeout") or 0) / n, 3)
            if n and not gaps else None)
        d["evidence_available"] = not gaps
        d["evidence_gaps"] = [g["reconciliation_id"] for g in gaps]
        d["reconciled_events"] = corrected
        d["excluded_events"] = len(excluded_ids)
        d["event_history_complete"] = len(effective) >= base_n
        return d


class SelfModel:
    """Competence store + live load probe + brief rendering.

    Load is read live (never cached) from the initiative store: intentions
    the mind has assigned and that are in progress. A missing store counts
    as zero.
    """

    def __init__(self, store: CompetenceStore, registry: Any = None, journal: Any = None) -> None:
        self.store = store
        self._registry = registry
        self.journal = journal      # ActionJournal (Amendment 1)
        self.perspective = None      # Optional source-backed working judgments; never authority.

    # -- recording (thin passthrough to the competence store) ------------
    def record(self, domain: str, outcome: str,
               latency_secs: Optional[float] = None,
               shadow: bool = False, violation: bool = False,
               stated_confidence: Optional[float] = None, *,
               source: Optional[str] = None,
               source_ref: Optional[str] = None,
               event_key: Optional[str] = None,
               evidence_status: str = "observed",
               outcome_contract: str = "legacy.unversioned",
               evidence: Optional[Dict[str, Any]] = None) -> bool:
        return self.store.record(domain, outcome, latency_secs=latency_secs,
                          shadow=shadow, violation=violation,
                          stated_confidence=stated_confidence,
                          source=source, source_ref=source_ref,
                          event_key=event_key,
                          evidence_status=evidence_status,
                          outcome_contract=outcome_contract,
                          evidence=evidence)

    # -- live load -------------------------------------------------------
    def load(self) -> Dict[str, int]:
        store = getattr(self._registry, "initiative_store", None) if self._registry is not None else None
        if store is None:
            try:
                from colonymind.api.routers.host import _initiative_store as store
            except Exception:
                store = None
        active = 0
        if store is not None and hasattr(store, "count"):
            try:
                active = int(store.count(status=["assigned", "acknowledged"]) or 0)
            except Exception:
                pass
        return {"active_initiatives": active, "total": active}

    def status(self) -> Dict[str, Any]:
        return {"domains": self.store.snapshot(), "load": self.load()}

    def brief(self) -> str:
        from colonymind.self_model.brief import self_brief
        parts = [self_brief(self.store.snapshot(), self.load())]
        if self.perspective is not None:
            parts.append(self.perspective.brief())
        return '\n'.join(part for part in parts if part)
