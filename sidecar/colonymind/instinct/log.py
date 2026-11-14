"""The decision log (``instinct_decisions`` in ``instinct.db``, in the instance's state directory), the per point
states and the local calibration records, each record ever put in force and each one displaced
(``instinct_record_history``, so the previous one, shipped or carried included, can be restored).

One row per decision: the event; the point and its version; per tier its state, raw and calibrated probabilities,
latency and outcome, and the calibration identity it was read under (provider kind, configured model, served
revision and the key's fingerprint); the final label, whether it was applied and the effect; the reason code;
references (ids only: a value that does not look like an id is kept as a keyed hash); and a keyed hash of the
state as sent (an HMAC under the instance's own secret in ``instinct.key``, so a short state cannot be guessed from
its hash, and the database alone cannot be used to test guesses). No text: no column holds words a point read. Rows older than
``RETENTION`` are dropped; what they counted is kept (``instinct_counts``). The outcome columns are filled in later.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from colonymind.util.temporal import now_utc

from .calibration import Record

logger = logging.getLogger(__name__)

DB_FILE = "instinct.db"
KEY_FILE = "instinct.key"          # the instance's own secret (``colonymind.instance.SECRET_FILE``)
#: How long a decision row is kept (my choice): long enough for the nightly refit and a season of outcomes.
RETENTION = timedelta(days=180)

SCHEMA = """
CREATE TABLE IF NOT EXISTS instinct_decisions (
  id TEXT PRIMARY KEY,
  at TEXT NOT NULL,
  event TEXT,
  point TEXT NOT NULL,
  version TEXT NOT NULL,
  tiers_json TEXT NOT NULL,
  label TEXT,
  applied INTEGER NOT NULL,
  effect TEXT,
  answered_by TEXT NOT NULL,
  reason TEXT NOT NULL,
  refs_json TEXT NOT NULL,
  state_hash TEXT,
  outcome_label TEXT,
  outcome_source TEXT,
  outcome_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_instinct_decisions_at ON instinct_decisions(at);
CREATE INDEX IF NOT EXISTS idx_instinct_decisions_point ON instinct_decisions(point, at);
CREATE TABLE IF NOT EXISTS instinct_counts (
  point TEXT NOT NULL, tier TEXT NOT NULL, outcome TEXT NOT NULL, n INTEGER NOT NULL,
  PRIMARY KEY (point, tier, outcome)
);
CREATE TABLE IF NOT EXISTS instinct_states (
  point TEXT NOT NULL, tier TEXT NOT NULL, state TEXT NOT NULL, at TEXT NOT NULL, source TEXT NOT NULL,
  evidence_json TEXT,
  PRIMARY KEY (point, tier)
);
CREATE TABLE IF NOT EXISTS instinct_records (
  fingerprint TEXT PRIMARY KEY, point TEXT NOT NULL, tier TEXT NOT NULL, record_json TEXT NOT NULL,
  at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS instinct_record_history (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, fingerprint TEXT NOT NULL, point TEXT NOT NULL,
  tier TEXT NOT NULL, record_json TEXT NOT NULL, at TEXT NOT NULL, reason TEXT NOT NULL, replaced TEXT
);
CREATE INDEX IF NOT EXISTS idx_instinct_record_history_id ON instinct_record_history(record_id);
CREATE TABLE IF NOT EXISTS instinct_outcomes (
  decision_id TEXT NOT NULL, point TEXT NOT NULL, resolver TEXT NOT NULL, source TEXT NOT NULL,
  verdict TEXT NOT NULL, judged TEXT, label TEXT, at TEXT NOT NULL, evidence_json TEXT,
  PRIMARY KEY (decision_id, resolver)
);
CREATE INDEX IF NOT EXISTS idx_instinct_outcomes_point ON instinct_outcomes(point, at);
CREATE TABLE IF NOT EXISTS instinct_transitions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, point TEXT NOT NULL, tier TEXT NOT NULL, from_state TEXT,
  to_state TEXT NOT NULL, at TEXT NOT NULL, source TEXT NOT NULL, evidence_json TEXT
);
CREATE TABLE IF NOT EXISTS instinct_examples (
  key TEXT NOT NULL, fingerprint TEXT NOT NULL, point TEXT NOT NULL, version TEXT NOT NULL, tier TEXT NOT NULL,
  kind TEXT NOT NULL, model TEXT NOT NULL, revision TEXT NOT NULL, raw_json TEXT NOT NULL, gold TEXT NOT NULL,
  existing TEXT, subset TEXT NOT NULL, tags_json TEXT, state_hash TEXT, set_name TEXT NOT NULL, at TEXT NOT NULL,
  group_key TEXT,
  PRIMARY KEY (key, fingerprint)
);
CREATE TABLE IF NOT EXISTS instinct_sets (
  name TEXT NOT NULL, fingerprint TEXT NOT NULL, point TEXT NOT NULL, set_hash TEXT, flip_rate REAL,
  items INTEGER NOT NULL, at TEXT NOT NULL, PRIMARY KEY (name, fingerprint, point)
);
CREATE TABLE IF NOT EXISTS instinct_audits (
  id TEXT PRIMARY KEY, at TEXT NOT NULL, point TEXT NOT NULL, tier TEXT NOT NULL, fingerprint TEXT,
  reason TEXT NOT NULL, evidence_json TEXT, closed_at TEXT
);
"""

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,127}$")
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MAX_REF_ITEMS = 16
_UNCONDITIONAL = object()
_ERASURE_HOOKS: Dict[Path, Callable[[Sequence[str]], Any]] = {}
_ERASURE_LOCK = threading.RLock()


def bind_erasure(path: Path, callback: Callable[[Sequence[str]], Any]) -> Callable[[], None]:
    """Bind a serialized erasure to every log handle for ``path``; return its unbind operation."""
    path = Path(path).resolve()
    with _ERASURE_LOCK:
        _ERASURE_HOOKS[path] = callback

    def unbind() -> None:
        with _ERASURE_LOCK:
            if _ERASURE_HOOKS.get(path) is callback:
                del _ERASURE_HOOKS[path]

    return unbind


def _now() -> datetime:
    return now_utc()


class Store:
    """The instance's ``instinct.db``; every write is its own short transaction on its own connection."""

    def __init__(self, path: Path, *, secret: Optional[bytes] = None, create: bool = True) -> None:
        self.path = Path(path)
        self._secret = secret
        self._lock = threading.Lock()
        self._ready = False
        if create:
            self._connect().close()

    # -- the database -------------------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        created = not self.path.exists()
        if created:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        if not self._ready:
            with self._lock:
                conn.executescript(SCHEMA)
                self._migrate(conn)
                conn.commit()
                self._ready = True
        if created:
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        return conn

    @classmethod
    def _migrate(cls, conn: sqlite3.Connection) -> None:
        """What was added since a table was first made: an imported answer's group (``group_key``), and a history
        entry for each record put in force before the history existed (so it can be restored)."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(instinct_examples)")}
        if "group_key" not in columns:
            conn.execute("ALTER TABLE instinct_examples ADD COLUMN group_key TEXT")
        for row in conn.execute("SELECT record_json, at FROM instinct_records").fetchall():
            try:
                record = Record.from_dict(json.loads(row["record_json"]))
            except (TypeError, ValueError):
                continue
            cls._archive(conn, record, reason="in force before the record history", at=row["at"])

    @staticmethod
    def _archive(conn: sqlite3.Connection, record: Record, *, reason: str, at: str) -> None:
        """Keep ``record`` in the history unless it is there already (a predecessor about to be displaced)."""
        if conn.execute("SELECT 1 FROM instinct_record_history WHERE record_id=?", (record.id,)).fetchone():
            return
        conn.execute("INSERT INTO instinct_record_history (record_id, fingerprint, point, tier, record_json, at, "
                     "reason, replaced) VALUES (?,?,?,?,?,?,?,NULL)",
                     (record.id, record.fingerprint, record.point, record.tier,
                      json.dumps(record.as_dict(), sort_keys=True), at, reason))

    # -- the instance secret --------------------------------------------------------------------------------

    def secret(self) -> bytes:
        """The instance's own secret beside the log (``colonymind.instance.instance_secret``), made on first use."""
        if self._secret is None:
            from colonymind.instance import instance_secret
            self._secret = instance_secret(self.path.parent)
        return self._secret

    def state_hash(self, text: str) -> str:
        return keyed_hash(self.secret(), text)

    # -- decisions --------------------------------------------------------------------------------------------

    def write(self, row: Mapping[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO instinct_decisions (id, at, event, point, version, tiers_json, label, applied, "
                "effect, answered_by, reason, refs_json, state_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["id"], row["at"], row.get("event") or "", row["point"], row["version"],
                 json.dumps(row.get("tiers") or {}, sort_keys=True), row.get("label"), 1 if row.get("applied") else 0,
                 row.get("effect") or "", row.get("answered_by") or "", row.get("reason") or "",
                 json.dumps(row.get("refs") or {}, sort_keys=True), row.get("state_hash")))

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        value["tiers"] = json.loads(value.pop("tiers_json") or "{}")
        value["refs"] = json.loads(value.pop("refs_json") or "{}")
        value["applied"] = bool(value["applied"])
        return value

    def rows(self, *, since: Optional[datetime] = None, point: Optional[str] = None,
             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Newest first."""
        if not self.path.exists():
            return []
        query, args = "SELECT * FROM instinct_decisions WHERE 1=1", []
        if since is not None:
            query += " AND at >= ?"
            args.append(since.isoformat())
        if point:
            query += " AND point = ?"
            args.append(point)
        query += " ORDER BY at DESC, rowid DESC"
        if limit is not None:
            query += " LIMIT ?"
            args.append(int(limit))
        with closing(self._connect()) as conn:
            return [self._row(row) for row in conn.execute(query, args).fetchall()]

    def prune(self, now: Optional[datetime] = None) -> int:
        """Drop rows older than ``RETENTION``, keeping what they counted; the number dropped."""
        cutoff = ((now or _now()) - RETENTION).isoformat()
        with closing(self._connect()) as conn, conn:
            old = conn.execute("SELECT id, point, tiers_json FROM instinct_decisions WHERE at < ?",
                               (cutoff,)).fetchall()
            self._fold(conn, old)
            conn.execute("DELETE FROM instinct_outcomes WHERE decision_id IN (SELECT id FROM instinct_decisions "
                         "WHERE at < ?)", (cutoff,))
            conn.execute("DELETE FROM instinct_decisions WHERE at < ?", (cutoff,))
        return len(old)

    def bind_erasure(self, callback: Callable[[Sequence[str]], Any]) -> Callable[[], None]:
        return bind_erasure(self.path, callback)

    def forget(self, ids: Sequence[str], *, reason: str = "erased",
               erasure: Optional[Callable[[Sequence[str]], Any]] = None) -> int:
        """Drop these decisions and their outcomes (their source was erased), keeping what they counted, and how
        many were dropped for ``reason`` per point (tier ``log``); the number dropped. The bound weight erasure
        finishes first; on failure the rows remain discoverable for a same-refs retry."""
        wanted = [str(ident) for ident in dict.fromkeys(ids) if ident]
        if not wanted:
            return 0
        with _ERASURE_LOCK:
            callback = erasure or _ERASURE_HOOKS.get(self.path.resolve())
        if callback is not None:
            result = callback(tuple(wanted))
        else:
            from ..plasticity.nightly import erase_for_log
            result = erase_for_log(self.path, tuple(wanted))
        clean = result.get("clean", True) if isinstance(result, Mapping) else getattr(result, "clean", True)
        if clean is False:
            raise ValueError("weight erasure is incomplete; retry with the same decision refs")
        dropped = 0
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA secure_delete=ON")
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                marks = ",".join("?" * len(chunk))
                rows = conn.execute(f"SELECT id, point, tiers_json FROM instinct_decisions WHERE id IN ({marks})",
                                    chunk).fetchall()
                self._fold(conn, rows)
                for row in rows:
                    conn.execute("INSERT INTO instinct_counts (point, tier, outcome, n) VALUES (?,?,?,1) ON CONFLICT"
                                 "(point, tier, outcome) DO UPDATE SET n = n + 1", (row["point"], "log", reason))
                conn.execute(f"DELETE FROM instinct_outcomes WHERE decision_id IN ({marks})", chunk)
                dropped += conn.execute(f"DELETE FROM instinct_decisions WHERE id IN ({marks})", chunk).rowcount
        return dropped

    @staticmethod
    def _fold(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> None:
        tally: Dict[Tuple[str, str, str], int] = {}
        for row in rows:
            for tier, entry in (json.loads(row["tiers_json"] or "{}") or {}).items():
                outcome = str((entry or {}).get("outcome") or "unknown")
                key = (row["point"], tier, outcome)
                tally[key] = tally.get(key, 0) + 1
        for (point, tier, outcome), n in tally.items():
            conn.execute("INSERT INTO instinct_counts (point, tier, outcome, n) VALUES (?,?,?,?) ON CONFLICT"
                         "(point, tier, outcome) DO UPDATE SET n = n + excluded.n", (point, tier, outcome, n))

    def row(self, ident: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with closing(self._connect()) as conn:
            found = conn.execute("SELECT * FROM instinct_decisions WHERE id=?", (str(ident),)).fetchone()
        return self._row(found) if found is not None else None

    # -- outcomes ---------------------------------------------------------------------------------------------

    def put_outcome(self, outcome: Mapping[str, Any], *, replace: bool = False) -> bool:
        """One resolver's outcome for a decision (once: a second from the same resolver is ignored, unless
        ``replace``: the owner's later word on the same decision), and the row's outcome columns set to its
        strongest outcome (``precedence``: the owner's own label first). False when the decision is gone or the
        resolver had already spoken."""
        with closing(self._connect()) as conn, conn:
            if conn.execute("SELECT 1 FROM instinct_decisions WHERE id=?", (outcome["decision"],)).fetchone() is None:
                return False
            written = conn.execute(
                "INSERT OR " + ("REPLACE" if replace else "IGNORE") + " INTO instinct_outcomes (decision_id, point, resolver, source, verdict, judged, label, "
                "at, evidence_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (outcome["decision"], outcome["point"], outcome["resolver"], outcome["source"], outcome["verdict"],
                 outcome.get("judged"), outcome.get("label"), outcome["at"],
                 json.dumps(dict(outcome.get("evidence") or {}), sort_keys=True))).rowcount
            best = self._strongest(conn, outcome["decision"])
            if best is not None:
                # The right label when it is known; else, for a wrong answer, the label it was not ("not:<label>").
                shown = best["label"] or (f"not:{best['judged']}" if best["verdict"] == "wrong" and best["judged"]
                                          else None)
                conn.execute("UPDATE instinct_decisions SET outcome_label=?, outcome_source=?, outcome_at=? WHERE id=?",
                             (shown, best["source"], best["at"], outcome["decision"]))
            return bool(written)

    def _strongest(self, conn: sqlite3.Connection, decision: str) -> Optional[Dict[str, Any]]:
        rows = [dict(row) for row in conn.execute("SELECT * FROM instinct_outcomes WHERE decision_id=?", (decision,))]
        rows.sort(key=precedence)
        return rows[0] if rows else None

    def outcomes(self, *, since: Optional[datetime] = None, point: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every outcome, oldest first, with its decision's id."""
        if not self.path.exists():
            return []
        query, args = "SELECT * FROM instinct_outcomes WHERE 1=1", []
        if since is not None:
            query += " AND at >= ?"
            args.append(since.isoformat())
        if point:
            query += " AND point = ?"
            args.append(point)
        query += " ORDER BY at, rowid"
        with closing(self._connect()) as conn:
            out = []
            for row in conn.execute(query, args):
                value = dict(row)
                value["evidence"] = json.loads(value.pop("evidence_json") or "{}")
                value["decision"] = value.pop("decision_id")
                out.append(value)
            return out

    def outcome_of(self, decision: str) -> Optional[Dict[str, Any]]:
        """A decision's strongest outcome, or None."""
        if not self.path.exists():
            return None
        with closing(self._connect()) as conn:
            best = self._strongest(conn, decision)
        if best is None:
            return None
        best["evidence"] = json.loads(best.pop("evidence_json") or "{}")
        best["decision"] = best.pop("decision_id")
        return best

    def kept_counts(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        out: Dict[str, Dict[str, Dict[str, int]]] = {}
        if not self.path.exists():
            return out
        with closing(self._connect()) as conn:
            for row in conn.execute("SELECT point, tier, outcome, n FROM instinct_counts"):
                out.setdefault(row["point"], {}).setdefault(row["tier"], {})[row["outcome"]] = int(row["n"])
        return out

    # -- states -----------------------------------------------------------------------------------------------

    def states(self) -> Dict[Tuple[str, str], str]:
        if not self.path.exists():
            return {}
        with closing(self._connect()) as conn:
            return {(row["point"], row["tier"]): row["state"]
                    for row in conn.execute("SELECT point, tier, state FROM instinct_states")}

    def set_state(self, point: str, tier: str, state: str, *, source: str,
                  evidence: Optional[Mapping[str, Any]] = None, at: Optional[datetime] = None,
                  expected_seq: Any = _UNCONDITIONAL) -> bool:
        """Write the state and transition together. With ``expected_seq``, write only if that is still the tier's
        newest transition (None means none existed); an intervening demotion therefore wins a carry."""
        when = (at or _now()).isoformat()
        body = json.dumps(dict(evidence or {}), sort_keys=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if expected_seq is not _UNCONDITIONAL:
                newest = conn.execute("SELECT MAX(seq) AS seq FROM instinct_transitions WHERE point=? AND tier=?",
                                      (point, tier)).fetchone()["seq"]
                if newest != expected_seq:
                    return False
            previous = conn.execute("SELECT state FROM instinct_states WHERE point=? AND tier=?",
                                    (point, tier)).fetchone()
            conn.execute("INSERT OR REPLACE INTO instinct_states (point, tier, state, at, source, evidence_json) "
                         "VALUES (?,?,?,?,?,?)", (point, tier, state, when, source, body))
            conn.execute("INSERT INTO instinct_transitions (point, tier, from_state, to_state, at, source, "
                         "evidence_json) VALUES (?,?,?,?,?,?,?)",
                         (point, tier, previous["state"] if previous is not None else None, state, when, source,
                          body))
        return True

    def transitions(self, *, point: Optional[str] = None, tier: Optional[str] = None,
                    limit: Optional[int] = 200, after_seq: Optional[int] = None,
                    since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Every state change, newest first: from, to, when, who made it and its evidence."""
        if not self.path.exists():
            return []
        query, args = "SELECT * FROM instinct_transitions WHERE 1=1", []
        for column, value in (("point", point), ("tier", tier)):
            if value:
                query += f" AND {column}=?"
                args.append(value)
        if after_seq is not None:
            query += " AND seq > ?"
            args.append(after_seq)
        if since is not None:
            query += " AND at >= ?"
            args.append(since.isoformat())
        query += " ORDER BY seq DESC"
        if limit is not None:
            query += " LIMIT ?"
            args.append(int(limit))
        with closing(self._connect()) as conn:
            out = []
            for row in conn.execute(query, args):
                value = dict(row)
                value["evidence"] = json.loads(value.pop("evidence_json") or "{}")
                out.append(value)
            return out

    # -- labelled sets (synthetic labels, by reference) --------------------------------------------------------

    def put_examples(self, examples: Sequence[Any], *, set_name: str, set_hash: str = "",
                     flip_rates: Optional[Mapping[str, float]] = None, at: Optional[datetime] = None) -> int:
        """Keep a labelled set's answers (no text: raw probabilities, labels, tags, a keyed hash of the state, the
        group whose variants share a fold) and, per point and fingerprint, the set's hash and its measured flip
        rate. The number kept."""
        when = (at or _now()).isoformat()
        seen: Dict[Tuple[str, str], int] = {}
        with closing(self._connect()) as conn, conn:
            for example in examples:
                conn.execute(
                    "INSERT OR REPLACE INTO instinct_examples (key, fingerprint, point, version, tier, kind, model, "
                    "revision, raw_json, gold, existing, subset, tags_json, state_hash, set_name, at, group_key) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (example.key, example.fingerprint, example.point, example.version, example.tier, example.kind,
                     example.model, example.revision, json.dumps(dict(example.raw), sort_keys=True), example.gold,
                     example.existing, example.subset, json.dumps(list(example.tags)), example.state_hash, set_name,
                     when, example.group or example.key))
                key = (example.point, example.fingerprint)
                seen[key] = seen.get(key, 0) + 1
            for (point, fingerprint), n in seen.items():
                rate = (flip_rates or {}).get(point)
                conn.execute("INSERT OR REPLACE INTO instinct_sets (name, fingerprint, point, set_hash, flip_rate, "
                             "items, at) VALUES (?,?,?,?,?,?,?)",
                             (set_name, fingerprint, point, set_hash, None if rate is None else float(rate), n, when))
        return sum(seen.values())

    def examples(self, *, fingerprint: Optional[str] = None, subset: Optional[str] = None,
                 point: Optional[str] = None) -> List[Any]:
        """The kept labelled answers, as ``fit.Example`` (source ``synthetic``)."""
        from .fit import Example
        if not self.path.exists():
            return []
        query, args = "SELECT * FROM instinct_examples WHERE 1=1", []
        for column, value in (("fingerprint", fingerprint), ("subset", subset), ("point", point)):
            if value:
                query += f" AND {column}=?"
                args.append(value)
        query += " ORDER BY set_name, key"
        with closing(self._connect()) as conn:
            return [Example(key=row["key"], point=row["point"], version=row["version"], tier=row["tier"],
                            kind=row["kind"], model=row["model"], revision=row["revision"],
                            raw=json.loads(row["raw_json"]), gold=row["gold"], existing=row["existing"],
                            source="synthetic", subset=row["subset"], tags=tuple(json.loads(row["tags_json"] or "[]")),
                            state_hash=row["state_hash"], group=row["group_key"] or row["key"])
                    for row in conn.execute(query, args)]

    def sets(self, *, fingerprint: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        query, args = "SELECT * FROM instinct_sets", []
        if fingerprint:
            query += " WHERE fingerprint=?"
            args.append(fingerprint)
        with closing(self._connect()) as conn:
            return [dict(row) for row in conn.execute(query + " ORDER BY at", args)]

    # -- local records ----------------------------------------------------------------------------------------

    def records(self) -> List[Record]:
        if not self.path.exists():
            return []
        with closing(self._connect()) as conn:
            out = []
            for row in conn.execute("SELECT record_json FROM instinct_records ORDER BY at"):
                try:
                    out.append(Record.from_dict(json.loads(row["record_json"])))
                except (TypeError, ValueError) as error:
                    logger.warning("an unreadable calibration record is ignored (%s)", type(error).__name__)
            return out

    def calibration_generation(self) -> bytes:
        """The erasure generation shared by readers in every process, with no record identities."""
        try:
            return self.path.with_name(self.path.name + ".calibration-generation").read_bytes()
        except FileNotFoundError:
            return b""

    def withdraw_revisions(self, revisions: Iterable[str], *, records: Iterable[str] = ()) -> int:
        """Remove active and archived calibrations for erased revisions, so restoration cannot revive them.
        Explicit record ids cover calibration identities a learner declares independently of its state digest."""
        revisions, record_ids = set(revisions), set(records)
        removed = set()
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA secure_delete=ON")
            for table, key in (("instinct_records", "fingerprint"), ("instinct_record_history", "seq")):
                for row in conn.execute(f"SELECT {key}, record_json FROM {table}").fetchall():
                    record = Record.from_dict(json.loads(row["record_json"]))
                    if record.id in record_ids or record.revision in revisions:
                        conn.execute(f"DELETE FROM {table} WHERE {key}=?", (row[key],))
                        removed.add(record.id)
        # Publish after the deletion commits, including on a retry that finds nothing left to delete.
        # Readers compare this token before reusing cached records, even in another process.
        from colonymind.plasticity._io import write_atomic
        write_atomic(self.path.with_name(self.path.name + ".calibration-generation"), uuid.uuid4().hex.encode())
        return len(removed)

    def put_record(self, record: Record, *, reason: str = "") -> None:
        """Put ``record`` in force for its fingerprint, and keep it in the history with what it replaced. The record
        it displaces is kept first when the history lacks it: the local one in force, else the shipped one for that
        fingerprint, so the first replacement can be undone like any other."""
        from . import records
        at = _now().isoformat()
        body = json.dumps(record.as_dict(), sort_keys=True)
        with closing(self._connect()) as conn, conn:
            previous = conn.execute("SELECT record_json, at FROM instinct_records WHERE fingerprint=?",
                                    (record.fingerprint,)).fetchone()
            replaced = None
            if previous is not None:
                try:
                    displaced = Record.from_dict(json.loads(previous["record_json"]))
                    self._archive(conn, displaced, reason="in force before the record history", at=previous["at"])
                    replaced = displaced.id
                except (TypeError, ValueError):
                    replaced = "unreadable"
            else:
                shipped = next((item for item in records.load() if item.fingerprint == record.fingerprint), None)
                if shipped is not None and shipped.id != record.id:
                    self._archive(conn, shipped, reason="shipped", at=at)
                    replaced = shipped.id
            conn.execute("INSERT OR REPLACE INTO instinct_records (fingerprint, point, tier, record_json, at) "
                         "VALUES (?,?,?,?,?)", (record.fingerprint, record.point, record.tier, body, at))
            conn.execute("INSERT INTO instinct_record_history (record_id, fingerprint, point, tier, record_json, at, "
                         "reason, replaced) VALUES (?,?,?,?,?,?,?,?)",
                         (record.id, record.fingerprint, record.point, record.tier, body, at,
                          str(reason or "put")[:200], replaced))

    def record_history(self, *, point: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """Every record put in force, newest first: ``{record, at, reason, replaced}``."""
        if not self.path.exists():
            return []
        query, args = "SELECT * FROM instinct_record_history", []
        if point:
            query += " WHERE point=?"
            args.append(point)
        query += " ORDER BY seq DESC LIMIT ?"
        args.append(int(limit))
        out = []
        with closing(self._connect()) as conn:
            for row in conn.execute(query, args):
                try:
                    record = Record.from_dict(json.loads(row["record_json"]))
                except (TypeError, ValueError):
                    continue
                out.append({"record": record, "at": row["at"], "reason": row["reason"], "replaced": row["replaced"]})
        return out

    def restore_record(self, record_id: str) -> Record:
        """Put a record from the history, or a shipped one, back in force (KeyError when neither has it)."""
        from . import records
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT record_json FROM instinct_record_history WHERE record_id=? ORDER BY seq DESC "
                               "LIMIT 1", (str(record_id),)).fetchone()
        if row is None:
            record = next((item for item in records.load() if item.id == str(record_id)), None)
            if record is None:
                raise KeyError(record_id)
        else:
            record = Record.from_dict(json.loads(row["record_json"]))
        self.put_record(record, reason=f"restored {record.id}")
        return record


    # -- audits ------------------------------------------------------------------------------------------------

    def open_audit(self, point: str, tier: str, fingerprint: str, reason: str, evidence: Mapping[str, Any], *,
                   at: Optional[datetime] = None) -> str:
        """An audit a person should read (a refit that saw what it must not): open until closed."""
        ident = f"aud-{uuid.uuid4().hex[:12]}"
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT INTO instinct_audits (id, at, point, tier, fingerprint, reason, evidence_json) "
                         "VALUES (?,?,?,?,?,?,?)", (ident, (at or _now()).isoformat(), point, tier, fingerprint,
                                                    str(reason)[:300], json.dumps(dict(evidence), sort_keys=True)))
        logger.warning("instinct audit opened for %s %s: %s", point, tier, reason)
        return ident

    def audits(self, *, open_only: bool = True) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        query = "SELECT * FROM instinct_audits" + (" WHERE closed_at IS NULL" if open_only else "") + " ORDER BY at"
        with closing(self._connect()) as conn:
            out = []
            for row in conn.execute(query):
                value = dict(row)
                value["evidence"] = json.loads(value.pop("evidence_json") or "{}")
                out.append(value)
            return out

#: The resolver name under which the owner's own label is kept (``outcomes.label``, ``colonymind instinct label``).
OWNER = "owner"
_RANK = {"explicit": 0, "synthetic": 1, "behavioural": 2, "interpreted": 3}


def precedence(outcome: Mapping[str, Any]) -> Tuple[int, int, float]:
    """An outcome's place among a decision's outcomes, strongest first: the owner's own label, then any other
    explicit outcome, then synthetic, then behavioural, then interpreted; within each, the latest supersedes the
    earlier. One rule for the row's outcome columns, the fit's labels and the tighten rule."""
    try:
        at = datetime.fromisoformat(str(outcome.get("at") or "").replace("Z", "+00:00")).timestamp()
    except ValueError:
        at = 0.0
    return (_RANK.get(str(outcome.get("source") or ""), 9), 0 if outcome.get("resolver") == OWNER else 1, -at)


def keyed_hash(secret: bytes, text: str) -> str:
    return hmac.new(secret, str(text).encode("utf-8"), hashlib.sha256).hexdigest()[:32]


class Hashed(str):
    """A reference kept only as a keyed hash, whatever it looks like: words that name something (an item's
    description) and could pass for an id when they are one word. The same words give the same hash, so a later
    outcome can find the decision."""


def references(refs: Optional[Mapping[str, Any]], secret: bytes) -> Dict[str, Any]:
    """References by id only: a key that is not a short code is dropped, and a value that does not look like an id
    (words, a long string), or is ``Hashed``, is kept as ``h:`` and a keyed hash of it."""
    out: Dict[str, Any] = {}
    for key, value in dict(refs or {}).items():
        if not isinstance(key, str) or not _KEY.match(key):
            continue
        if isinstance(value, (list, tuple)):
            out[key] = [_ref(item, secret) for item in list(value)[:MAX_REF_ITEMS]]
        elif value is not None and value != "":
            out[key] = _ref(value, secret)
    return out


def _ref(value: Any, secret: bytes) -> str:
    text = str(value)
    return text if _ID.match(text) and not isinstance(value, Hashed) else "h:" + keyed_hash(secret, text)[:24]


def identifier(value: Any, secret: bytes) -> str:
    """One identifier as the log keeps it (a model name, a served revision): as it is when it looks like an id,
    ``h:`` and a keyed hash of it otherwise, "" for none."""
    return "" if value is None or value == "" else _ref(value, secret)


def code(value: Any) -> str:
    """A short code (an event name) or ""."""
    text = str(value or "")
    return text if _CODE.match(text) else ""


__all__ = ["DB_FILE", "KEY_FILE", "OWNER", "RETENTION", "SCHEMA", "Hashed", "Store", "bind_erasure", "code", "identifier",
           "keyed_hash", "precedence", "references"]
