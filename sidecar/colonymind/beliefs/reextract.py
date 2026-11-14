"""Re-extraction: a person's history formed into claims again under the current extraction version.

Sources are never backfilled and a stored claim keeps the version that formed it (docs/SOURCE-CLAIMS.md), so a
better extractor helps only new messages. ``colonymind memory reextract`` queues the attributed person-scope sources
whose claims an older ``EXTRACTION_VERSION`` formed; the source worker's ``reextract`` lane extracts them again.

Order. One person at a time, each person's sources in the order they were said (``OCCURRENCE_SQL``, then
``ingested_at`` and ``turn_id``), however the queue was filled: a source waits while an earlier one of the same
person is pending, so a correction is always read after the message it corrects. Each message is offered what it
was offered the first time: the records of sources said before it and of its own source's earlier messages (never a
later message's), with the records its own earlier claims retired offered live again, and the key list of those
records led by its own earlier claims' keys.

Commit. A message's new claims go through the same extraction, validation, review and commit as a new message's
(``SourceClaimProjection._commit``). In one transaction per message, its earlier claims leave the store (a claim
another message also states stays, anchored there), the new claims commit and what was decided about the earlier
ones carries forward. A failure anywhere (the model, the output, the commit, a claim of the new reading the commit
rejects) rolls the transaction back: the message keeps its earlier claims. A message that kept them is not recorded
done, and its source is not marked current (its job ends ``kept``): the next run queues the source again and reads
only that message. An earlier claim leaves the store only in
the transaction that commits its successor; it is kept, with its link ``reextracted_by`` to the claim that holds its
place (none when the new reading has no counterpart), in ``source_claim_reextractions``, which erasure reaches like
the claims. A corrected or superseded claim the new reading has no counterpart for stays as it was, at its message:
recall covers its words with its correction, and a value read anew elsewhere is retired by it.

Carry-forward. The same reading of the same message is the same claim: its id, and with it every reference to it
(an owner's answer, a held change, a relation, a lesson), stays. A claim that changed (a new key or wording) has a
successor, chosen by the same id, then the same key and value, then the same subject and value under another
predicate, then the same key (the last two only when unambiguous, ``_successors``): the successor takes the old
claim's retraction or supersession, the links that pointed at the old claim, the claims whose subject or words it
grounds, the held changes that name it, its learning tier, and the running mind's decided relations when the value
is the same. A retracted value is never made live again:
- a retraction or supersession the new reading lacks is carried from the old claim;
- a value the message now states for the first time, which the store retracted or superseded after the message was
  said, is retracted or superseded with it;
- a restatement folds into a claim that was retired only after the message was said, as it did when it was said,
  and that claim is retired again exactly as it was;
- a link the new reading would add to retire a claim the owner kept (their answer to a contradiction question, a
  held change they ratified or declined) is not applied: it is reported;
- a held preference change the owner already decided is decided the same way, never asked again;
- a message whose earlier claim grounds another claim that cannot follow the new reading keeps its earlier claims.
What is reported (``source_reextract_reports``) holds ids only.

Imported history and checkpoints are never queued (they have no claim job: quotations only). Erased sources and
sources whose attribution changed are skipped. The lane yields: it starts nothing while a live recall runs or a new
turn's extraction is due, waits for live recall between two messages, and holds a message's commit while a new
turn of the same person is being extracted (that turn may hold one of the message's claims as its prior). It is
resumable: each message's commit records it done, under the current version, so a stopped job goes on with its next
message, and a later run keeps a source's checkpoint (after an upgrade of the extractor, its done messages are read
again).
"""
from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import dataclass
import json
import logging
import math
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from .source_claims import (extract_claims, extraction_diagnostics, norm_value,
                            projection_timeout_seconds)

logger = logging.getLogger(__name__)

# How often one source is tried before it is left failed with its earlier claims (a builder's choice, not a measured
# bound): enough to ride out a model restart, few enough that a message the model cannot read does not hold the
# person's later sources for long. A missing extraction role is not an attempt: the source waits for it.
MAX_ATTEMPTS = 5
LEASE_SECONDS = 60.0
# How long a step waits for a live recall to end before it yields its turn (the source-vector lane's bound).
YIELD_SECONDS = 8.0
# The newest finished re-extraction jobs, and claim jobs, the measured rates read (a builder's choice).
RATE_SAMPLE = 200
_NOT_EXTRACTED = frozenset({"unsupported_message", "oversize_message"})


def _version() -> str:
    """The version new claims record: the commit's own (``source_projection.EXTRACTION_VERSION``)."""
    from . import source_projection
    return source_projection.EXTRACTION_VERSION


def _occurrence(alias: str = "s") -> str:
    from .source_projection import OCCURRENCE_SQL
    return OCCURRENCE_SQL.format(s=alias)


def initialize(conn) -> None:
    """The queue, the runs, the retired claims and the reports, beside the claims in the source ledger."""
    conn.execute('''CREATE TABLE IF NOT EXISTS source_reextract_runs (
        id TEXT PRIMARY KEY, scope TEXT NOT NULL, person_id TEXT, target_version TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'running', limit_count INTEGER, queued INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, finished_at REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS source_reextract_jobs (
        turn_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, contact_id TEXT NOT NULL, occurred_jd REAL NOT NULL,
        ingested_at TEXT NOT NULL, from_version TEXT, status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
        lease_until REAL NOT NULL DEFAULT 0, lease_token TEXT NOT NULL DEFAULT '', done_json TEXT NOT NULL DEFAULT '[]',
        done_version TEXT, error TEXT, outcome_json TEXT, enqueued_at REAL NOT NULL, finished_at REAL)''')
    if 'done_version' not in {row[1] for row in conn.execute('PRAGMA table_info(source_reextract_jobs)')}:
        # A queue an earlier build created: its checkpoints name no version, so they count for nothing (``_done``)
        # and those messages are read again under the current one.
        conn.execute('ALTER TABLE source_reextract_jobs ADD COLUMN done_version TEXT')
    conn.execute('CREATE INDEX IF NOT EXISTS source_reextract_order '
                 'ON source_reextract_jobs(contact_id,status,occurred_jd)')
    # The lane looks for waiting work every idle pause; finished rows stay as the record of what was re-extracted.
    conn.execute('CREATE INDEX IF NOT EXISTS source_reextract_status ON source_reextract_jobs(status)')
    conn.execute('''CREATE TABLE IF NOT EXISTS source_claim_reextractions (
        claim_id TEXT PRIMARY KEY, reextracted_by TEXT, run_id TEXT, turn_id TEXT NOT NULL, message_hash TEXT NOT NULL,
        subject_key TEXT NOT NULL, predicate TEXT NOT NULL, value_key TEXT NOT NULL, data_json TEXT NOT NULL,
        valid_from TEXT, valid_to TEXT, superseded_by TEXT, retracted_by TEXT, retired_at REAL NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_reextraction_turn ON source_claim_reextractions(turn_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS source_claim_reextraction_by '
                 'ON source_claim_reextractions(reextracted_by)')
    conn.execute('''CREATE TABLE IF NOT EXISTS source_reextract_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, turn_id TEXT NOT NULL, message_hash TEXT,
        kind TEXT NOT NULL, claim_id TEXT, other_id TEXT, at REAL NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS source_reextract_report_turn ON source_reextract_reports(turn_id)')


def erase_removed(conn, turn_id: str, retained_hashes: Iterable[str], *, whole: bool,
                  roots: Iterable[str] = ()) -> None:
    """Erasure reaches what re-extraction keeps of a message: its retired claims (their words) and its reports.
    ``roots``: the live claims anchored at the erased messages, read before they leave. A retired claim goes too when
    it carries the subject or the words (``subject_basis_claim_id``, ``value_parts``) of an erased claim, live or
    retired, however many retired or live claims lie between; a live claim that carries them only through a retired
    one goes with it, as ``value_revision.erase_dependents`` removes the live ones. A retired claim whose subject an
    erased message named (``subject_basis_message``) goes too. A source erased whole leaves the queue."""
    if not _has_table(conn, "source_claim_reextractions"):
        return
    kept = list(retained_hashes)
    marks = ",".join("?" for _ in kept)
    clause = f" AND message_hash NOT IN ({marks})" if kept else ""
    roots = list(roots)
    here = [row[0] for row in conn.execute("SELECT claim_id FROM source_claim_reextractions WHERE turn_id=?" + clause,
                                           (turn_id, *kept))]
    # A retired claim whose subject an erased message named carries that message's words.
    here += [row[0] for row in conn.execute(
        "SELECT claim_id,json_extract(data_json,'$.subject_basis_message.message_hash') "
        "FROM source_claim_reextractions WHERE json_extract(data_json,'$.subject_basis_message.turn_id')=?",
        (turn_id,)) if row[1] not in kept]
    if (roots or here) and conn.execute("SELECT 1 FROM source_claim_reextractions LIMIT 1").fetchone():
        removed = [row[0] for row in conn.execute("""WITH RECURSIVE claim(id, data_json) AS (
                SELECT id, data_json FROM source_claims
                UNION ALL SELECT claim_id, data_json FROM source_claim_reextractions),
            removed(id) AS (SELECT value FROM json_each(?) UNION SELECT c.id FROM claim c JOIN removed r
                ON json_extract(c.data_json,'$.subject_basis_claim_id')=r.id OR EXISTS
                    (SELECT 1 FROM json_each(c.data_json,'$.value_parts') p
                     WHERE json_extract(p.value,'$.source_claim_id')=r.id))
            SELECT id FROM removed""", (json.dumps(list(dict.fromkeys([*roots, *here]))),))]
        # The roots themselves are the live erasure's (one may stay, anchored at another witness).
        own = set(roots)
        removed = [ident for ident in removed if ident not in own]
        for chunk in _chunks(removed):
            places = ",".join("?" for _ in chunk)
            conn.execute(f"DELETE FROM source_claim_reextractions WHERE claim_id IN ({places})", chunk)
            conn.execute(f"DELETE FROM source_claims WHERE id IN ({places})", chunk)
            conn.execute(f"DELETE FROM source_claim_witnesses WHERE claim_id IN ({places})", chunk)
    conn.execute("DELETE FROM source_claim_reextractions WHERE turn_id=?" + clause, (turn_id, *kept))
    conn.execute("DELETE FROM source_reextract_reports WHERE turn_id=?" + clause, (turn_id, *kept))
    if whole:
        conn.execute("DELETE FROM source_reextract_jobs WHERE turn_id=?", (turn_id,))


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _checkpoint(conn) -> str:
    """The queue's checkpoint columns as a read selects them (``q``): a queue an earlier build created has no
    ``done_version`` until the ledger is opened for writing (``initialize``), and until then its checkpoints name no
    version, so they count for nothing (``_done``). A read-only connection (``--status``, ``--dry-run``) reads it
    so, and a start plans so before it initializes the queue."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(source_reextract_jobs)")}
    return "q.done_json," + ("q.done_version" if "done_version" in columns else "NULL AS done_version")


# -- the plan: what a start would queue, and what it would cost ------------------------------------------------------

def _eligible_where(conn, person: Optional[str], *, queued: bool) -> tuple:
    """The attributed person-scope sources whose claims an older version formed: those no run is working yet, or
    (``queued``) those one is. Imported history and checkpoints have no claim job; erased and reattributed sources
    are left."""
    where = ["j.status='complete'", "coalesce(j.extraction_version,'')<>?", "s.scope='person'",
             "NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)",
             "NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)"]
    args: List[Any] = [_version()]
    if person:
        where.append("s.contact_id=?")
        args.append(person)
    working = ("EXISTS (SELECT 1 FROM source_reextract_jobs q WHERE q.turn_id=s.turn_id "
               "AND q.status IN ('pending','running'))")
    if _has_table(conn, "source_reextract_jobs"):
        where.append(working if queued else "NOT " + working)
    elif queued:
        where.append("0")
    return " AND ".join(where), args


def _eligible(conn, *, person: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    """What a start queues and a plan describes, the same selection for both: the eligible sources no run is working
    yet (``_eligible_where``), one person at a time in the order they were said, the first ``limit``. Each carries
    the checkpoint an earlier run left for it (``done_json``, ``done_version``: none before any run): a source a run
    finished with a message that kept its earlier claims is eligible again, and only that message is read again."""
    where, args = _eligible_where(conn, person, queued=False)
    checkpoint = (_checkpoint(conn) + " FROM source_claim_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id "
                  "LEFT JOIN source_reextract_jobs q ON q.turn_id=s.turn_id"
                  if _has_table(conn, "source_reextract_jobs") else
                  "NULL AS done_json,NULL AS done_version FROM source_claim_jobs j JOIN turn_sources s "
                  "ON s.turn_id=j.turn_id")
    query = (f"SELECT s.turn_id,s.contact_id,s.session_id,s.messages_json,s.ingested_at,{_occurrence()} AS occurred_jd,"
             "j.extraction_version," + checkpoint + " WHERE "
             + where + f" ORDER BY s.contact_id,{_occurrence()},s.ingested_at,s.turn_id")
    from colonymind.memory import curated
    selected = []
    for row in conn.execute(query, args):
        if row['occurred_jd'] is None or curated.is_forgotten(conn, row['turn_id']):
            continue
        selected.append(dict(row))
        if limit is not None and len(selected) >= int(limit):
            break
    return selected


def _messages(source: Dict[str, Any]) -> Dict[str, int]:
    """A source's user messages still to be read (those its checkpoint lists under the current version are read
    already, ``_done``), and of them those the extraction call reads (``_extract_claims``'s own bounds)."""
    from colonymind.turns.audio import claim_message
    from colonymind.turns.idempotency import source_message_hash
    counts = {"messages": 0, "model_messages": 0}
    done = set(_done(source)) if source.get("done_json") is not None else set()
    for raw in json.loads(source["messages_json"]):
        view = claim_message(raw)
        content = view.get("content") if view else None
        if not view or view.get("role") != "user" or not isinstance(content, str) or not content.strip():
            continue
        if done and source_message_hash(source["session_id"], view) in done:
            continue
        counts["messages"] += 1
        counts["model_messages"] += int(len(content) <= 12000)
    return counts


def _chunks(values: List[Any], size: int = 500):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def measured_rate(conn) -> Optional[Dict[str, Any]]:
    """What one message cost, measured by the re-extraction this instance already ran (its newest ``RATE_SAMPLE``
    finished sources): model seconds, calls and tokens a message. None before any."""
    if not _has_table(conn, "source_reextract_jobs"):
        return None
    totals = {"messages": 0, "seconds": 0.0, "calls": 0, "tokens": 0, "token_messages": 0}
    for (encoded,) in conn.execute("SELECT outcome_json FROM source_reextract_jobs WHERE outcome_json IS NOT NULL "
                                   "AND status IN ('complete','kept','pending','running','failed','cancelled') "
                                   "ORDER BY coalesce(finished_at,enqueued_at) DESC LIMIT ?", (RATE_SAMPLE,)):
        outcome = json.loads(encoded or "{}")
        totals["messages"] += int(outcome.get("messages") or 0)
        totals["seconds"] += float(outcome.get("seconds") or 0)
        totals["calls"] += int(outcome.get("calls") or 0)
        if outcome.get("tokens") is not None:
            totals["tokens"] += int(outcome["tokens"])
            totals["token_messages"] += int(outcome.get("messages") or 0)
    if not totals["messages"]:
        return None
    return {"messages": totals["messages"], "seconds_per_message": totals["seconds"] / totals["messages"],
            "calls_per_message": totals["calls"] / totals["messages"],
            "tokens_per_message": (round(totals["tokens"] / totals["token_messages"])
                                   if totals["token_messages"] else None)}


def review_share(conn) -> Optional[Dict[str, Any]]:
    """Review calls per extraction call over the newest finished claim jobs (their diagnostics): the other half of
    what a message costs before any re-extraction has been measured."""
    extractions = reviews = jobs = 0
    for (encoded,) in conn.execute("SELECT diagnostics_json FROM source_claim_jobs WHERE status='complete' "
                                   "AND diagnostics_json IS NOT NULL ORDER BY rowid DESC LIMIT ?", (RATE_SAMPLE,)):
        try:
            diagnostics = json.loads(encoded)
        except ValueError:
            continue
        extractions += int(diagnostics.get("response_count") or 0)
        reviews += int(diagnostics.get("review_response_count") or 0)
        jobs += 1
    if not extractions:
        return None
    return {"share": reviews / extractions, "jobs": jobs}


def plan(conn, *, person: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """What ``start`` would queue now (the same selection, ``_eligible``) and what it would cost; reads only (a
    read-only connection will do). ``queued``: the eligible sources of the scope a run already works, apart.

    ``estimate``: model calls, and model seconds and tokens at the rate this instance measured (``measured_rate``;
    None before a re-extraction has run: a small run with ``limit`` measures it). Seconds are model time: the lane
    yields to live work, so the elapsed time is longer."""
    sources = _eligible(conn, person=person, limit=limit)
    turns = [row["turn_id"] for row in sources]
    counts = {"messages": 0, "model_messages": 0}
    versions: Dict[str, int] = {}
    for source in sources:
        for key, value in _messages(source).items():
            counts[key] += value
        version = source["extraction_version"] or "unknown"
        versions[version] = versions.get(version, 0) + 1
    current = _version()
    claims = 0
    for chunk in _chunks(turns):
        marks = ",".join("?" for _ in chunk)
        claims += conn.execute(f"SELECT count(*) FROM source_claims WHERE turn_id IN ({marks}) "
                               "AND coalesce(json_extract(data_json,'$.extraction_version'),'')<>?",
                               (*chunk, current)).fetchone()[0]
    where, args = _eligible_where(conn, person, queued=True)
    queued = conn.execute("SELECT count(*) FROM source_claim_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id "
                          "WHERE " + where, args).fetchone()[0]
    rate = measured_rate(conn)
    share = review_share(conn)
    model_messages = counts["model_messages"]
    if rate is not None:
        calls = round(model_messages * rate["calls_per_message"])
        seconds = model_messages * rate["seconds_per_message"]
        tokens = (model_messages * rate["tokens_per_message"] if rate["tokens_per_message"] is not None else None)
    else:
        calls = round(model_messages * (1 + (share["share"] if share else 1.0)))
        seconds = tokens = None
    return {"version": current, "scope": "person" if person else "all", "person": person, "limit": limit,
            "people": len({row["contact_id"] for row in sources}), "sources": len(sources), **counts,
            "claims": claims, "from_versions": versions, "queued": queued, "rate": rate, "review_share": share,
            "estimate": {"calls": calls, "seconds": seconds, "tokens": tokens}}


def start(conn, *, person: Optional[str] = None, limit: Optional[int] = None,
          now: Optional[float] = None) -> Dict[str, Any]:
    """Queue a run: the sources ``plan`` names (no run is working them yet). Returns the run id (None when there
    was nothing to queue) and how many sources it queued."""
    initialize(conn)
    now = time.time() if now is None else now
    conn.execute("BEGIN IMMEDIATE")
    try:
        sources = _eligible(conn, person=person, limit=limit)
        if not sources:
            conn.execute("COMMIT")
            return {"run_id": None, "queued": 0}
        run_id = uuid.uuid4().hex
        conn.execute("INSERT INTO source_reextract_runs(id,scope,person_id,target_version,limit_count,queued,"
                     "created_at) VALUES (?,?,?,?,?,?,?)",
                     (run_id, "person" if person else "all", person, _version(), limit, len(sources), now))
        for source in sources:
            # A source an earlier run left keeps its checkpoint: the messages it read under the current version
            # are not read again (``_done``), only the one that kept its earlier claims.
            conn.execute('''INSERT INTO source_reextract_jobs(turn_id,run_id,contact_id,occurred_jd,ingested_at,
                from_version,status,enqueued_at) VALUES (?,?,?,?,?,?,'pending',?)
                ON CONFLICT(turn_id) DO UPDATE SET run_id=excluded.run_id,contact_id=excluded.contact_id,
                occurred_jd=excluded.occurred_jd,ingested_at=excluded.ingested_at,from_version=excluded.from_version,
                status='pending',attempts=0,next_attempt=0,lease_until=0,lease_token='',error=NULL,
                outcome_json=NULL,enqueued_at=excluded.enqueued_at,finished_at=NULL''',
                (source["turn_id"], run_id, source["contact_id"], source["occurred_jd"], source["ingested_at"],
                 source["extraction_version"], now))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return {"run_id": run_id, "queued": len(sources)}


def status(conn, *, runs: int = 10) -> Dict[str, Any]:
    """The newest runs with their sources by state and their reports by kind, the measured rate, and the model
    messages still waiting (and their model time at that rate)."""
    result: Dict[str, Any] = {"version": _version(), "runs": [], "rate": measured_rate(conn),
                              "remaining_messages": 0, "remaining_seconds": None}
    if not _has_table(conn, "source_reextract_runs"):
        return result
    for run in conn.execute("SELECT * FROM source_reextract_runs ORDER BY created_at DESC LIMIT ?", (runs,)):
        jobs = {row[0]: row[1] for row in conn.execute(
            "SELECT status,count(*) FROM source_reextract_jobs WHERE run_id=? GROUP BY status", (run["id"],))}
        reports = {row[0]: row[1] for row in conn.execute(
            "SELECT kind,count(*) FROM source_reextract_reports WHERE run_id=? GROUP BY kind", (run["id"],))}
        result["runs"].append({"id": run["id"], "scope": run["scope"], "person": run["person_id"],
                               "version": run["target_version"], "status": run["status"],
                               "created_at": run["created_at"], "finished_at": run["finished_at"],
                               "limit": run["limit_count"], "jobs": jobs, "reports": reports})
    waiting = [dict(row) for row in conn.execute(
        f"SELECT s.session_id,s.messages_json,{_checkpoint(conn)} FROM source_reextract_jobs q "
        "JOIN turn_sources s ON s.turn_id=q.turn_id WHERE q.status IN ('pending','running')")]
    for row in waiting:
        result["remaining_messages"] += _messages(row)["model_messages"]
    if result["rate"] is not None:
        result["remaining_seconds"] = result["remaining_messages"] * result["rate"]["seconds_per_message"]
    return result


def _done(job) -> List[str]:
    """The messages of a queued source already read again, under the current version only: a checkpoint written
    under an older version (the extractor was upgraded while the run waited) counts for nothing, so those messages
    are read again and the source is never marked current with claims an older version formed."""
    if job["done_version"] != _version():
        return []
    return json.loads(job["done_json"] or "[]")


def cancel(conn, *, now: Optional[float] = None) -> Dict[str, Any]:
    """Stop what waits: every pending or running source is cancelled (a running one stops before its next message
    commits) and keeps the claims it has. Returns how many were cancelled."""
    if not _has_table(conn, "source_reextract_jobs"):
        return {"cancelled": 0}
    now = time.time() if now is None else now
    conn.execute("BEGIN IMMEDIATE")
    try:
        cancelled = conn.execute("UPDATE source_reextract_jobs SET status='cancelled',lease_until=0,finished_at=? "
                                 "WHERE status IN ('pending','running')", (now,)).rowcount
        conn.execute("UPDATE source_reextract_runs SET status='cancelled',finished_at=? WHERE status='running'", (now,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return {"cancelled": cancelled}


# -- the worker ----------------------------------------------------------------------------------------------------

class _Keep(Exception):
    """This message keeps its earlier claims (the transaction's changes for it are rolled back), for ``kind``.
    ``rejected``: the commit's rejections (``(stage, reason)``), logged again once the rollback has undone them."""

    def __init__(self, kind: str, claim_id: Optional[str] = None, other_id: Optional[str] = None, *,
                 rejected: Iterable[tuple] = ()):
        super().__init__(kind)
        self.kind, self.claim_id, self.other_id = kind, claim_id, other_id
        self.rejected = list(rejected)


@dataclass
class _Old:
    """One of a message's earlier claims: a claim anchored at the message, or a claim anchored elsewhere that the
    message witnessed (``anchored`` False)."""
    id: str
    subject_key: str
    predicate: str
    value_key: str
    representation: str
    data: Dict[str, Any]
    retracted_by: Optional[str]
    superseded_by: Optional[str]
    valid_from: Optional[str]
    valid_to: Optional[str]
    anchored: bool
    archived: bool = False

    @property
    def live(self) -> bool:
        return not self.retracted_by and not self.superseded_by


@dataclass
class _New:
    """A claim the commit made of the message's new reading: ``own`` when it is a new row anchored at the message,
    else the existing claim it restates (or that the message already minted)."""
    id: str
    subject_key: str
    predicate: str
    value_key: str
    representation: str
    own: bool


def _representation(data: Dict[str, Any]) -> str:
    from .source_projection import _representation as representation
    return representation(data)


class Reextraction:
    """The ``reextract`` lane of the source worker, and the plan, start, status and cancel of the ledger it serves.

    ``relations``: a callable that returns the running mind's state (``claims.relation:`` entries) or None; a
    decided relation follows a claim to its successor with the same value."""

    def __init__(self, projection, *, relations=None):
        self.projection = projection
        self.ledger = projection.ledger
        self.relations = relations

    # -- the ledger's plan, start, status and cancel --

    def plan(self, *, person: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        with closing(self.ledger._connect()) as conn:
            return plan(conn, person=person, limit=limit)

    def start(self, *, person: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        with closing(self.ledger._connect()) as conn:
            return start(conn, person=person, limit=limit)

    def status(self) -> Dict[str, Any]:
        with closing(self.ledger._connect()) as conn:
            return status(conn)

    def cancel(self) -> Dict[str, Any]:
        with closing(self.ledger._connect()) as conn:
            return cancel(conn)

    # -- the lane --

    async def process_one(self, router) -> bool:
        """One source: True when it did work (a source finished, deferred or skipped), False when nothing is due or
        the lane yielded."""
        if not self._waiting():
            return False
        if not await self._quiet():
            return False
        job = self._claim()
        if job is None:
            return False
        try:
            await self._work(job, router)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._defer(job, type(exc).__name__)
            logger.warning("re-extraction of a source deferred (%s)", type(exc).__name__)
        return True

    def _waiting(self) -> bool:
        with closing(self.ledger._connect()) as conn:
            return conn.execute("SELECT 1 FROM source_reextract_jobs WHERE status IN ('pending','running') "
                                "LIMIT 1").fetchone() is not None

    def _live_extraction(self) -> bool:
        """A new turn's claim job is due or in hand: it goes first (the backlog's rule, ``projection_backlog``)."""
        from colonymind.turns import projection_backlog
        now = time.time()
        with closing(self.ledger._connect()) as conn:
            return conn.execute("SELECT 1 FROM source_claim_jobs WHERE enqueued_at>=? AND ((status='pending' AND "
                                "next_attempt<=?) OR (status='running' AND lease_until>?)) LIMIT 1",
                                (projection_backlog.watermark(conn), now, now)).fetchone() is not None

    async def _quiet(self) -> bool:
        from colonymind.memory.live_recall import LIVE
        if LIVE.active() and not await LIVE.quiet(YIELD_SECONDS):
            return False
        return not self._live_extraction()

    def _claim(self) -> Optional[Dict[str, Any]]:
        """Lease the next source: for some person, their earliest source still waiting, when it is due."""
        now = time.time()
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            from colonymind.memory import curated
            waiting = conn.execute('''SELECT j.* FROM source_reextract_jobs j
                WHERE j.status IN ('pending','running')
                ORDER BY j.contact_id,j.occurred_jd,j.ingested_at,j.turn_id''').fetchall()
            seen, row = set(), None
            for candidate in waiting:
                if curated.is_forgotten(self.ledger, candidate['turn_id'], conn=conn):
                    continue
                person = candidate['contact_id']
                if person in seen:
                    continue
                seen.add(person)
                if ((candidate['status'] == 'pending' and candidate['next_attempt'] <= now)
                        or (candidate['status'] == 'running' and candidate['lease_until'] <= now)):
                    row = candidate
                    break
            if row is None:
                return None
            token = uuid.uuid4().hex
            conn.execute("UPDATE source_reextract_jobs SET status='running',attempts=attempts+1,lease_until=?,"
                         "lease_token=? WHERE turn_id=?", (now + LEASE_SECONDS, token, row["turn_id"]))
            return dict(row, lease_token=token, attempts=row["attempts"] + 1)

    def _renew(self, job, request_timeout: float) -> bool:
        with closing(self.ledger._connect()) as conn, conn:
            return conn.execute("UPDATE source_reextract_jobs SET lease_until=? WHERE turn_id=? AND status='running' "
                                "AND lease_token=?", (time.time() + request_timeout + 30, job["turn_id"],
                                                      job["lease_token"])).rowcount == 1

    def _defer(self, job, error: str, *, attempt: bool = True) -> None:
        """The source waits and keeps its claims: again after a backoff, or left failed after ``MAX_ATTEMPTS``."""
        now = time.time()
        attempts = job["attempts"] if attempt else job["attempts"] - 1
        with closing(self.ledger._connect()) as conn, conn:
            if attempts >= MAX_ATTEMPTS:
                conn.execute("UPDATE source_reextract_jobs SET status='failed',error=?,lease_until=0,finished_at=? "
                             "WHERE turn_id=? AND lease_token=? AND status='running'",
                             (error, now, job["turn_id"], job["lease_token"]))
                self._settle_run(conn, job["run_id"], now)
            else:
                conn.execute("UPDATE source_reextract_jobs SET status='pending',error=?,attempts=?,lease_until=0,"
                             "next_attempt=? WHERE turn_id=? AND lease_token=? AND status='running'",
                             (error, attempts, now + min(900, 15 * 2 ** min(attempts, 6)), job["turn_id"],
                              job["lease_token"]))

    @staticmethod
    def _settle_run(conn, run_id: str, now: float) -> None:
        if not conn.execute("SELECT 1 FROM source_reextract_jobs WHERE run_id=? AND status IN ('pending','running') "
                            "LIMIT 1", (run_id,)).fetchone():
            conn.execute("UPDATE source_reextract_runs SET status='complete',finished_at=? WHERE id=? "
                         "AND status='running'", (now, run_id))

    def _end(self, job, status: str, *, model: Optional[str] = None) -> None:
        """The source is done (``complete``), done but for a message that kept its earlier claims (``kept``), or has
        nothing left to read (``skipped``). Only a complete source's claim job records the current version, so no
        later run queues it again; a kept one still holds claims an older version formed, so the next run queues it
        again and reads only the kept message (its checkpoint lists the others)."""
        now = time.time()
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            ended = conn.execute("UPDATE source_reextract_jobs SET status=?,error=NULL,lease_until=0,finished_at=? "
                                 "WHERE turn_id=? AND lease_token=? AND status='running'",
                                 (status, now, job["turn_id"], job["lease_token"])).rowcount
            if ended and status == "complete":
                conn.execute("UPDATE source_claim_jobs SET extraction_version=?,model=coalesce(?,model) "
                             "WHERE turn_id=? AND status='complete'", (_version(), model, job["turn_id"]))
            if ended:
                self._settle_run(conn, job["run_id"], now)

    def _source(self, job) -> Optional[Dict[str, Any]]:
        """The job's source as it stands, while it is still an attributed person-scope source of the same person."""
        with closing(self.ledger._connect()) as conn:
            row = conn.execute('''SELECT s.*,j.timezone FROM turn_sources s
                JOIN source_claim_jobs j ON j.turn_id=s.turn_id
                WHERE s.turn_id=? AND s.contact_id=? AND s.scope='person' AND j.status='complete'
                AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
                AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)''',
                (job["turn_id"], job["contact_id"])).fetchone()
        return dict(row) if row else None

    async def _work(self, job, router) -> None:
        from colonymind.turns.audio import claim_message
        from colonymind.turns.idempotency import source_message_hash
        from colonymind.memory.live_recall import LIVE
        from .source_projection import KNOWN_KEYS, _Metered
        source = self._source(job)
        if source is None:
            self._end(job, "skipped")
            return
        done = set(_done(job))
        # Each message is offered the records of the sources said before its own, and of its own source's messages
        # said before it (read again already, as they were read first the first time).
        earlier: List[str] = []
        model = None
        first = True
        kept = False
        for raw in json.loads(source["messages_json"]):
            view = claim_message(raw)
            content = view.get("content") if view else None
            if not view or view.get("role") != "user" or not isinstance(content, str) or not content.strip():
                continue
            message_hash = source_message_hash(source["session_id"], view)
            before = (job["occurred_jd"], job["ingested_at"], job["turn_id"], tuple(earlier))
            earlier.append(message_hash)
            if message_hash in done:
                continue
            if not first and LIVE.active():
                await LIVE.quiet(YIELD_SECONDS)       # between two messages: wait for live recall, then go on
            first = False
            request_timeout = projection_timeout_seconds(router)
            if not self._renew(job, request_timeout):
                return                                # cancelled, or taken over after a lost lease
            old = self._earlier(source, message_hash)
            reviving = {row.id for row in old if row.anchored}
            prior = await self.projection.prior_claims(source, view, before=before, reviving=reviving)
            # The key list: the message's own earlier keys first, so a property keeps its name, then the keys of
            # the records said before it.
            keys: Dict[tuple, Dict[str, str]] = {}
            for row in old:
                if row.representation != "episode" and isinstance(row.data.get("subject"), str):
                    keys.setdefault((row.data["subject"].casefold(), row.predicate),
                                    {"subject": row.data["subject"], "predicate": row.predicate})
            for key in self.projection.known_keys(source, before=before):
                keys.setdefault((str(key["subject"]).casefold(), key["predicate"]), key)
            metered = _Metered(router)
            started = time.monotonic()
            claims, model = await extract_claims(metered, source, view, prior, timezone_name=source["timezone"],
                                                 request_timeout=request_timeout,
                                                 diagnostics=extraction_diagnostics(),
                                                 known_keys=list(keys.values())[:KNOWN_KEYS],
                                                 earlier=self.projection.earlier_messages(source, view))
            if model == "local_extraction_role_unavailable":
                self._defer(job, model, attempt=False)
                return
            measured = {"messages": 1, "seconds": time.monotonic() - started, "calls": metered.calls,
                        "tokens": metered.tokens.get("total_tokens", sum(
                            metered.tokens.get(key, 0) for key in ("prompt_tokens", "completion_tokens")))
                        if metered.tokens else None}
            if model in _NOT_EXTRACTED:
                claims, measured = None, {"messages": 0}
            applied = self._apply(job, view, message_hash, claims, model, measured)
            if applied == "lost":
                return
            if applied == "busy":
                self._defer(job, "live_extraction", attempt=False)
                return
            if applied == "gone":
                self._end(job, "skipped")
                return
            kept = kept or applied == "kept"
        self._end(job, "kept" if kept else "complete", model=model if model not in _NOT_EXTRACTED else None)

    def _earlier(self, source, message_hash: str, conn=None) -> List[_Old]:
        """The message's claims an older version formed: anchored at it, or witnessed by it."""
        current = _version()
        if conn is None:
            with closing(self.ledger._connect()) as own:
                return self._earlier(source, message_hash, own)
        found = []
        for row in conn.execute("SELECT * FROM source_claims WHERE turn_id=? AND message_hash=? AND "
                                "coalesce(json_extract(data_json,'$.extraction_version'),'')<>? ORDER BY id",
                                (source["turn_id"], message_hash, current)):
            data = json.loads(row["data_json"])
            found.append(_Old(row["id"], row["subject_key"], row["predicate"], row["value_key"], _representation(data),
                              data, row["retracted_by"], row["superseded_by"], row["valid_from"], row["valid_to"],
                              anchored=True))
        for row in conn.execute('''SELECT c.*,w.data_json AS witness_json FROM source_claim_witnesses w
                JOIN source_claims c ON c.id=w.claim_id WHERE w.turn_id=? AND w.message_hash=?
                AND coalesce(json_extract(w.data_json,'$.extraction_version'),'')<>? ORDER BY c.id''',
                                (source["turn_id"], message_hash, current)):
            data = json.loads(row["witness_json"])
            found.append(_Old(row["id"], row["subject_key"], row["predicate"], row["value_key"], _representation(data),
                              data, row["retracted_by"], row["superseded_by"], row["valid_from"], row["valid_to"],
                              anchored=False))
        return found

    def _apply(self, job, view, message_hash: str, claims, model, measured) -> str:
        """One message, in one transaction: its new reading committed in place of its earlier claims, and the
        message recorded done. ``claims`` None: nothing to read (the message keeps its claims). ``kept``: the new
        reading could not replace them (``_Keep``); the message is not recorded done, so it is read again."""
        relations: List[tuple] = []
        kept = False
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute("SELECT 1 FROM source_reextract_jobs WHERE turn_id=? AND status='running' "
                                "AND lease_token=?", (job["turn_id"], job["lease_token"])).fetchone():
                return "lost"
            current = conn.execute('''SELECT s.* FROM turn_sources s WHERE s.turn_id=? AND s.contact_id=?
                AND s.scope='person'
                AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
                AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)''',
                (job["turn_id"], job["contact_id"])).fetchone()
            if current is None:
                return "gone"
            # A new turn of the same person being extracted now may have been offered one of this message's claims
            # as its prior: this message waits for its commit, so the link it makes is never to a retired claim.
            if conn.execute("SELECT 1 FROM source_claim_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id "
                            "WHERE s.contact_id=? AND j.turn_id<>? AND j.status='running' AND j.lease_until>? "
                            "LIMIT 1", (job["contact_id"], job["turn_id"], time.time())).fetchone():
                return "busy"
            from colonymind.turns.audio import claim_message
            from colonymind.turns.idempotency import source_message_hash
            present = any(source_message_hash(current["session_id"], claim_message(raw) or raw) == message_hash
                          for raw in json.loads(current["messages_json"]))
            effects: Dict[str, int] = {}
            if present and claims is not None:
                conn.execute("SAVEPOINT reextract_message")
                try:
                    effects, relations = self._replace(conn, dict(current), view, message_hash, claims, model, job)
                except _Keep as keep:
                    conn.execute("ROLLBACK TO reextract_message")
                    self._report(conn, job, message_hash, keep.kind, keep.claim_id, keep.other_id)
                    if keep.rejected:
                        from colonymind.memory import learning
                        for stage, reason in keep.rejected:
                            learning.log(conn, store="claim", action="rejected", stage=stage or "commit",
                                         reason=reason or "", contact_id=current["contact_id"],
                                         refs=[current["turn_id"], message_hash])
                    effects = {"kept": 1}
                    kept = True
                conn.execute("RELEASE reextract_message")
            elif claims is None and self._earlier(dict(current), message_hash, conn):
                self._report(conn, job, message_hash, "not_extracted")
            row = conn.execute("SELECT done_json,done_version,outcome_json FROM source_reextract_jobs WHERE turn_id=?",
                               (job["turn_id"],)).fetchone()
            done = _done(row) + ([] if kept else [message_hash])
            outcome = json.loads(row["outcome_json"] or "{}")
            for key, value in {**measured, **effects}.items():
                if value is None:
                    continue
                outcome[key] = (outcome.get(key) or 0) + value
            conn.execute("UPDATE source_reextract_jobs SET done_json=?,done_version=?,outcome_json=? WHERE turn_id=?",
                         (json.dumps(done), _version(), json.dumps(outcome, sort_keys=True), job["turn_id"]))
        if relations:
            self._carry_relations(relations)
        return "kept" if kept else "applied"

    @staticmethod
    def _report(conn, job, message_hash, kind, claim_id=None, other_id=None) -> None:
        conn.execute("INSERT INTO source_reextract_reports(run_id,turn_id,message_hash,kind,claim_id,other_id,at) "
                     "VALUES (?,?,?,?,?,?,?)", (job["run_id"], job["turn_id"], message_hash, kind, claim_id, other_id,
                                                time.time()))

    # -- the replacement of one message's claims --------------------------------------------------------------------

    def _replace(self, conn, current, view, message_hash, claims, model, job):
        """Returns the effects (counts) and the pairs of an earlier claim and its successor with the same value
        (for the mind's relations, carried once the transaction commits)."""
        from .source_projection import drop_witness, promote_witness
        from colonymind.turns import claim_keys
        contact, turn = current["contact_id"], current["turn_id"]
        old = self._earlier(current, message_hash, conn)
        anchored = [row for row in old if row.anchored]
        own_ids = {row.id for row in anchored}
        live_before = {(row.subject_key, row.predicate, row.value_key) for row in old if row.live}

        # The records the message's earlier claims retired stand as they did before it, so its new reading can
        # retire them again through the commit. Their end of validity stays.
        unretired: Dict[str, Dict[str, Any]] = {}
        if own_ids:
            marks = ",".join("?" for _ in own_ids)
            for row in conn.execute(f"SELECT id,retracted_by,superseded_by,valid_to FROM source_claims WHERE "
                                    f"retracted_by IN ({marks}) OR superseded_by IN ({marks})",
                                    (*own_ids, *own_ids)).fetchall():
                unretired[row["id"]] = dict(row)
                conn.execute("UPDATE source_claims SET retracted_by=NULL,superseded_by=NULL WHERE id=?", (row["id"],))

        # The earlier view leaves: a witness edge goes; an anchored claim another message states moves there;
        # the rest is retired into the archive (its link is written once its successor is known). What a retired
        # (corrected or superseded) claim was is kept aside: with no counterpart in the new reading it comes back.
        kept_aside: Dict[str, tuple] = {}
        for row in old:
            if not row.anchored:
                if not row.live:
                    edge = conn.execute("SELECT * FROM source_claim_witnesses WHERE claim_id=? AND turn_id=? AND "
                                        "message_hash=?", (row.id, turn, message_hash)).fetchone()
                    count = conn.execute("SELECT witnesses FROM source_claims WHERE id=?", (row.id,)).fetchone()
                    if edge is not None and count is not None:
                        kept_aside[row.id] = (dict(edge), [], count[0])
                drop_witness(conn, row.id, turn, message_hash)
        for row in anchored:
            if not row.live:
                kept_aside[row.id] = (dict(conn.execute("SELECT * FROM source_claims WHERE id=?", (row.id,))
                                          .fetchone()),
                                      [dict(edge) for edge in conn.execute(
                                          "SELECT * FROM source_claim_witnesses WHERE claim_id=?", (row.id,))], None)
            if conn.execute("SELECT 1 FROM source_claim_witnesses WHERE claim_id=? LIMIT 1", (row.id,)).fetchone() \
                    and promote_witness(conn, row.id):
                continue
            raw = conn.execute("SELECT * FROM source_claims WHERE id=?", (row.id,)).fetchone()
            conn.execute('''INSERT OR REPLACE INTO source_claim_reextractions(claim_id,reextracted_by,run_id,turn_id,
                message_hash,subject_key,predicate,value_key,data_json,valid_from,valid_to,superseded_by,retracted_by,
                retired_at) VALUES (?,NULL,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (row.id, job["run_id"], raw["turn_id"], raw["message_hash"], raw["subject_key"], raw["predicate"],
                 raw["value_key"], raw["data_json"], raw["valid_from"], raw["valid_to"], row.superseded_by,
                 row.retracted_by, time.time()))
            conn.execute("DELETE FROM source_claim_witnesses WHERE claim_id=?", (row.id,))
            conn.execute("DELETE FROM source_claims WHERE id=?", (row.id,))
            row.archived = True

        # A claim of the new reading's key and value that was retired after the message was said stood live when
        # the message was said: the commit sees it so (a restatement folds into it, as it did then), and it is
        # retired again, exactly as it was, once the commit is done.
        later = self._as_then(conn, contact, claims, job["occurred_jd"], set(unretired))

        # What the commit may retire, as it stands now: a refused link restores it exactly.
        standing: Dict[str, Optional[str]] = {}
        for claim in claims:
            prior = claim.get("prior_claim_id")
            key = conn.execute("SELECT subject_key,predicate,value_key FROM source_claims WHERE id=?",
                               (prior,)).fetchone() if prior else None
            if key is not None:
                for peer in conn.execute('''SELECT c.id,c.valid_to FROM source_claims c JOIN turn_sources s
                        ON s.turn_id=c.turn_id WHERE s.contact_id=? AND c.subject_key=? AND c.predicate=?
                        AND c.value_key=?''', (contact, *key)):
                    standing[peer["id"]] = peer["valid_to"]
        pending_before = {row[0] for row in conn.execute(
            "SELECT id FROM learning_pending WHERE contact_id=? AND status='pending'", (contact,))}

        outcomes: List[Dict[str, Any]] = []
        self.projection._commit(conn, {"turn_id": turn, "session_id": current["session_id"]}, view, claims,
                                model=model, outcomes=outcomes)
        # The whole new reading commits, or none of it: a claim the commit rejects (a basis that stopped being
        # current while the message was read) would leave an earlier claim without the claim that holds its place.
        rejected = [outcome for outcome in outcomes if outcome.get("verdict") == "rejected"]
        if rejected or len(outcomes) < len(claims):
            raise _Keep("commit_rejected", rejected=[(outcome.get("stage"), outcome.get("reason"))
                                                     for outcome in rejected])
        for ident, before in later.items():
            conn.execute("UPDATE source_claims SET retracted_by=?,superseded_by=?,valid_to=? WHERE id=?",
                         (before["retracted_by"], before["superseded_by"], before["valid_to"], ident))
        new: List[_New] = []
        for claim, outcome in zip(claims, outcomes):
            ident = outcome["claim_id"]
            if outcome.get("effect") == "new":
                new.append(_New(ident, claim["subject_key"], claim["predicate"], norm_value(claim["value"]),
                                _representation(claim), own=True))
                continue
            if outcome["verdict"] == "unchanged":
                self._witness(conn, ident, current, message_hash, claim, model)
            row = conn.execute("SELECT subject_key,predicate,value_key,data_json FROM source_claims WHERE id=?",
                               (ident,)).fetchone()
            if row is not None:
                new.append(_New(ident, row["subject_key"], row["predicate"], row["value_key"],
                                _representation(json.loads(row["data_json"])), own=False))
        successor = self._successors(old, new)
        effects = {"new": sum(item.own for item in new), "retired": 0, "carried": 0, "reported": 0}

        # A retired claim the new reading has no counterpart for stays as it was, at this message: its correction
        # is what recall shows for the words it covers (recall offers uncovered words as a plain quotation), and
        # what a value read anew from another message is retired by. The retirement of a claim it retired is
        # restored below with the rest.
        restored = set()
        for row in old:
            if row.live or row.id in successor or row.id not in kept_aside:
                continue
            record, edges, count = kept_aside[row.id]
            if row.anchored:
                # Archived, or moved to another message that states it: it comes back as it was, anchored here.
                conn.execute("DELETE FROM source_claim_witnesses WHERE claim_id=?", (row.id,))
                conn.execute("DELETE FROM source_claims WHERE id=?", (row.id,))
                self._insert(conn, "source_claims", record)
                for edge in edges:
                    self._insert(conn, "source_claim_witnesses", edge)
                conn.execute("DELETE FROM source_claim_reextractions WHERE claim_id=?", (row.id,))
                row.archived = False
            elif conn.execute("SELECT 1 FROM source_claims WHERE id=?", (row.id,)).fetchone():
                self._insert(conn, "source_claim_witnesses", record)
                conn.execute("UPDATE source_claims SET witnesses=? WHERE id=?", (count, row.id))
            restored.add(row.id)

        def report(kind, claim_id=None, other_id=None):
            self._report(conn, job, message_hash, kind, claim_id, other_id)
            effects["reported"] += 1

        # Each earlier claim's place.
        same_value: List[tuple] = []
        for row in old:
            if row.id in restored:
                continue
            match = successor.get(row.id)
            following = match[0] if match else None
            if following is not None and following.id != row.id and match[1] <= 2:
                same_value.append((row.id, following.id))
            if not row.live:
                if following is not None and following.own:
                    effects["carried"] += self._take_lifecycle(conn, following.id, row, report)
                elif following is not None and following.id != row.id and self._live(conn, following.id):
                    report("link_contradicted", row.id, following.id)
            if not row.archived:
                continue
            if following is not None and following.id == row.id:
                conn.execute("DELETE FROM source_claim_reextractions WHERE claim_id=?", (row.id,))
                continue
            effects["retired"] += 1
            if following is not None:
                conn.execute("UPDATE source_claim_reextractions SET reextracted_by=? WHERE claim_id=?",
                             (following.id, row.id))
                conn.execute("UPDATE learning_pending SET by_id=? WHERE by_id=? AND status='pending'",
                             (following.id, row.id))
                conn.execute("UPDATE learning_pending SET target_id=? WHERE target_id=? AND status='pending'",
                             (following.id, row.id))
                conn.execute('''INSERT OR IGNORE INTO learning_items(store,item_id,tier,tier_since,useful,last_useful_at)
                    SELECT store,?,tier,tier_since,useful,last_useful_at FROM learning_items
                    WHERE store='claim' AND item_id=?''', (following.id, row.id))
            self._follow_dependents(conn, contact, row, following)
        # A claim the new reading kept under its id keeps what grounds on it only while that still holds.
        for row in anchored:
            match = successor.get(row.id)
            if row.archived and match and match[0].id == row.id:
                self._follow_dependents(conn, contact, row, match[0])

        # The records the earlier claims retired: retired again by the new reading, else by the successor of the
        # claim that retired them (or that claim's own id: never revived).
        for ident, before in unretired.items():
            now = conn.execute("SELECT retracted_by,superseded_by FROM source_claims WHERE id=?", (ident,)).fetchone()
            if now is None or now["retracted_by"] or now["superseded_by"]:
                continue
            by = before["retracted_by"] or before["superseded_by"]
            origin = next(row for row in anchored if row.id == by)
            match = successor.get(by)
            holder = match[0].id if match and match[0].own else by
            column = "retracted_by" if before["retracted_by"] else "superseded_by"
            conn.execute(f"UPDATE source_claims SET {column}=?,valid_to=? WHERE id=?", (holder, before["valid_to"], ident))
            effects["carried"] += 1
            if match and match[0].own:
                self._take_link(conn, holder, origin, ident)

        # A link the new reading adds may not retire what the owner kept.
        own_new = [item.id for item in new if item.own]
        if own_new:
            marks = ",".join("?" for _ in own_new)
            for target in conn.execute(f"SELECT id,retracted_by,superseded_by FROM source_claims WHERE "
                                       f"retracted_by IN ({marks}) OR superseded_by IN ({marks})",
                                       (*own_new, *own_new)).fetchall():
                if target["id"] in unretired or not self._owner_kept(conn, target["id"]):
                    continue
                by = target["retracted_by"] or target["superseded_by"]
                conn.execute("UPDATE source_claims SET retracted_by=NULL,superseded_by=NULL,valid_to=? WHERE id=?",
                             (standing.get(target["id"]), target["id"]))
                self._drop_link(conn, by, target["id"])
                report("owner_kept", target["id"], by)
        # A value this message states for the first time stays retired where the store retired it after the
        # message was said.
        for item in new:
            if not item.own or (item.subject_key, item.predicate, item.value_key) in live_before:
                continue
            effects["carried"] += self._keep_retired(conn, contact, item, job["occurred_jd"], own_ids)
        # A held change the owner already decided is decided again the same way.
        for pending in conn.execute("SELECT * FROM learning_pending WHERE contact_id=? AND status='pending'",
                                    (contact,)).fetchall():
            if pending["id"] in pending_before:
                continue
            effects["carried"] += self._decided(conn, dict(pending), old, successor, pending_before)

        claim_keys.index_pending(conn)
        return effects, same_value

    @staticmethod
    def _insert(conn, table: str, record: Dict[str, Any]) -> None:
        columns = list(record)
        conn.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                     [record[column] for column in columns])

    @staticmethod
    def _holder(conn, ident: Optional[str]) -> Optional[str]:
        """The claim that holds ``ident``'s place now: ``ident`` while it is in the store, else the claim a
        re-extraction retired it for (followed to the end); an id that is no claim (an owner's answer) is itself."""
        seen = set()
        while ident and ident not in seen:
            seen.add(ident)
            if conn.execute("SELECT 1 FROM source_claims WHERE id=?", (ident,)).fetchone():
                return ident
            row = conn.execute("SELECT reextracted_by FROM source_claim_reextractions WHERE claim_id=?",
                               (ident,)).fetchone()
            if row is None or not row[0]:
                return ident
            ident = row[0]
        return ident

    @staticmethod
    def _live(conn, ident: str) -> bool:
        row = conn.execute("SELECT retracted_by,superseded_by FROM source_claims WHERE id=?", (ident,)).fetchone()
        return row is not None and not row["retracted_by"] and not row["superseded_by"]

    def _witness(self, conn, ident, current, message_hash, claim, model) -> None:
        """The new reading is a claim this message minted, now anchored at another message: the message witnesses it
        again, as the fold records a restatement (the newest witness is the anchor)."""
        anchor = conn.execute("SELECT c.*,s.ingested_at FROM source_claims c JOIN turn_sources s "
                              "ON s.turn_id=c.turn_id WHERE c.id=?", (ident,)).fetchone()
        if anchor is None or (anchor["turn_id"], anchor["message_hash"]) == (current["turn_id"], message_hash) or \
                conn.execute("SELECT 1 FROM source_claim_witnesses WHERE claim_id=? AND turn_id=? AND message_hash=?",
                             (ident, current["turn_id"], message_hash)).fetchone():
            return
        data = json.dumps(dict(claim, model=model, extraction_version=_version(), role="user"), ensure_ascii=False)
        if (current["ingested_at"] or "") >= (anchor["ingested_at"] or ""):
            conn.execute("INSERT OR IGNORE INTO source_claim_witnesses(claim_id,turn_id,message_hash,data_json,"
                         "recorded_at) VALUES (?,?,?,?,?)", (ident, anchor["turn_id"], anchor["message_hash"],
                                                             anchor["data_json"], anchor["ingested_at"]))
            conn.execute("UPDATE source_claims SET turn_id=?,message_hash=?,data_json=?,witnesses=witnesses+1 "
                         "WHERE id=?", (current["turn_id"], message_hash, data, ident))
        else:
            conn.execute("INSERT INTO source_claim_witnesses(claim_id,turn_id,message_hash,data_json,recorded_at) "
                         "VALUES (?,?,?,?,?)", (ident, current["turn_id"], message_hash, data,
                                                current["ingested_at"]))
            conn.execute("UPDATE source_claims SET witnesses=witnesses+1 WHERE id=?", (ident,))

    @staticmethod
    def _successors(old: List[_Old], new: List[_New]) -> Dict[str, tuple]:
        """Each earlier claim's successor among the new reading's claims, and how it was matched: 0 the same id,
        1 the same key and value, 2 the same subject and value under another predicate, 3 the same key. None when
        nothing matches.

        The first two levels are the same claim read again: an unused claim is preferred, and several earlier
        claims may share one. The last two are a guess, and the successor takes the earlier claim's correction,
        links and the owner's decisions, so a guess must be unambiguous: a claim of another subject never matches,
        and a match holds only when the earlier claim has exactly one unused candidate at that level and no other
        earlier claim has it as a candidate there."""
        levels = (
            lambda o, n: n.id == o.id,
            lambda o, n: (n.subject_key, n.predicate, n.value_key) == (o.subject_key, o.predicate, o.value_key),
            lambda o, n: (n.subject_key, n.value_key, n.representation) == (o.subject_key, o.value_key,
                                                                            o.representation),
            lambda o, n: (n.subject_key, n.predicate) == (o.subject_key, o.predicate),
        )
        result: Dict[str, tuple] = {}
        used: set = set()
        for level, matches in enumerate(levels):
            if level < 2:
                for row in old:
                    if row.id in result:
                        continue
                    found = next((item for item in new if item.id not in used and matches(row, item)), None) or \
                        next((item for item in new if matches(row, item)), None)
                    if found is not None:
                        result[row.id] = (found, level)
                        used.add(found.id)
                continue
            waiting = [row for row in old if row.id not in result]
            candidates = {row.id: [item for item in new if item.id not in used and matches(row, item)]
                          for row in waiting}
            claimed: Dict[str, int] = {}
            for found in candidates.values():
                for item in found:
                    claimed[item.id] = claimed.get(item.id, 0) + 1
            for row in waiting:
                found = candidates[row.id]
                if len(found) == 1 and claimed[found[0].id] == 1:
                    result[row.id] = (found[0], level)
                    used.add(found[0].id)
        return result

    @staticmethod
    def _take_lifecycle(conn, ident: str, old: _Old, report) -> int:
        """The successor takes the earlier claim's retraction or supersession, when it is live and the change is
        still later than it; a supersession it would outlast is reported instead."""
        row = conn.execute("SELECT retracted_by,superseded_by,valid_from,valid_to FROM source_claims WHERE id=?",
                           (ident,)).fetchone()
        if row is None or row["retracted_by"] or row["superseded_by"]:
            return 0
        if old.retracted_by:
            conn.execute("UPDATE source_claims SET retracted_by=? WHERE id=?", (old.retracted_by, ident))
            return 1
        if old.valid_to and row["valid_from"] and row["valid_from"] >= old.valid_to:
            report("link_contradicted", old.id, ident)
            return 0
        valid_to = min(filter(None, (row["valid_to"], old.valid_to)), default=None)
        conn.execute("UPDATE source_claims SET superseded_by=?,valid_to=? WHERE id=?", (old.superseded_by, valid_to,
                                                                                        ident))
        return 1

    @staticmethod
    def _take_link(conn, holder: str, origin: _Old, target: str) -> None:
        """The claim that now retires ``target`` in the earlier claim's place says so, when its own reading named no
        record (a correction whose link the new reading missed stays a correction of it)."""
        row = conn.execute("SELECT data_json,valid_from FROM source_claims WHERE id=?", (holder,)).fetchone()
        data = json.loads(row["data_json"])
        if data.get("prior_claim_id") or data.get("operation") in {"correct", "change"}:
            return
        data.update(operation=origin.data.get("operation") or "assert",
                    prior_claim_id=origin.data.get("prior_claim_id") or target)
        valid_from = row["valid_from"]
        if data["operation"] == "change" and valid_from is None and origin.valid_from:
            valid_from = data["valid_from"] = origin.valid_from
            data["validity_basis"] = origin.data.get("validity_basis") or data.get("validity_basis")
        conn.execute("UPDATE source_claims SET data_json=?,valid_from=? WHERE id=?",
                     (json.dumps(data, ensure_ascii=False), valid_from, holder))

    @staticmethod
    def _drop_link(conn, holder: Optional[str], target: str) -> None:
        """A refused link leaves no trace in the claim that proposed it: it stands as a plain statement."""
        if not holder:
            return
        row = conn.execute("SELECT data_json FROM source_claims WHERE id=?", (holder,)).fetchone()
        if row is None:
            return
        data = json.loads(row["data_json"])
        if data.get("prior_claim_id") == target:
            data.update(operation="assert", prior_claim_id=None)
            conn.execute("UPDATE source_claims SET data_json=? WHERE id=?", (json.dumps(data, ensure_ascii=False),
                                                                             holder))

    def _follow_dependents(self, conn, contact: str, old: _Old, following: Optional[_New]) -> None:
        """The claims whose subject or carried words an earlier claim grounds follow its successor; when one cannot
        (no successor, or one that no longer grounds it), the message keeps its earlier claims (``_Keep``)."""
        # A basis is always the same person's claim (``subject_basis``, ``value_basis``): read that person's only.
        dependents = [dict(row) for row in conn.execute('''SELECT c.id,c.data_json FROM source_claims c
            JOIN turn_sources s ON s.turn_id=c.turn_id WHERE s.contact_id=? AND
            (json_extract(c.data_json,'$.subject_basis_claim_id')=? OR EXISTS (SELECT 1 FROM
            json_each(c.data_json,'$.value_parts') p WHERE json_extract(p.value,'$.source_claim_id')=?))''',
            (contact, old.id, old.id))]
        if not dependents:
            return
        if following is None:
            raise _Keep("kept_for_dependents", old.id)
        for dependent in dependents:
            if following.id != old.id:
                data = json.loads(dependent["data_json"])
                if data.get("subject_basis_claim_id") == old.id:
                    data["subject_basis_claim_id"] = following.id
                for part in data.get("value_parts") or []:
                    if part.get("source_claim_id") == old.id:
                        part["source_claim_id"] = following.id
                conn.execute("UPDATE source_claims SET data_json=? WHERE id=?",
                             (json.dumps(data, ensure_ascii=False), dependent["id"]))
            if not self.projection._rows(conn, contact, "", ids=[dependent["id"]]):
                raise _Keep("kept_for_dependents", old.id, dependent["id"])

    @staticmethod
    def _owner_kept(conn, ident: str) -> bool:
        """The owner kept this claim, or a claim it re-extracted: their answer to a contradiction question kept it,
        they ratified the change it made, or they declined a change that would retire it."""
        seen, frontier = {ident}, [ident]
        while frontier:
            marks = ",".join("?" for _ in frontier)
            if conn.execute(f"SELECT 1 FROM learning_log WHERE store='claim' AND action='ratified' "
                            f"AND stage='owner_answer' AND item_id IN ({marks}) LIMIT 1", frontier).fetchone():
                return True
            if conn.execute(f"SELECT 1 FROM learning_pending WHERE (status='ratified' AND by_id IN ({marks})) OR "
                            f"(status='declined' AND target_id IN ({marks})) LIMIT 1", (*frontier, *frontier)
                            ).fetchone():
                return True
            earlier = [row[0] for row in conn.execute(
                f"SELECT claim_id FROM source_claim_reextractions WHERE reextracted_by IN ({marks})", frontier)]
            frontier = [item for item in earlier if item not in seen]
            seen.update(frontier)
        return False

    def _as_then(self, conn, contact: str, claims, occurred_jd: float, skip: set) -> Dict[str, Dict[str, Any]]:
        """The retired claims with a new claim's key and value whose retirement came after the message was said,
        made live for the commit; returns what to restore. A supersession's end of validity is lifted with it when
        it is the superseding claim's start (what the supersession wrote)."""
        later: Dict[str, Dict[str, Any]] = {}
        for claim in claims:
            key = (claim["subject_key"], claim["predicate"], norm_value(claim["value"]))
            for row in conn.execute('''SELECT c.id,c.retracted_by,c.superseded_by,c.valid_to FROM source_claims c
                    JOIN turn_sources s ON s.turn_id=c.turn_id WHERE s.contact_id=? AND s.scope='person'
                    AND c.subject_key=? AND c.predicate=? AND c.value_key=?
                    AND (c.retracted_by IS NOT NULL OR c.superseded_by IS NOT NULL)''', (contact, *key)).fetchall():
                if row["id"] in skip or row["id"] in later:
                    continue
                by = row["retracted_by"] or row["superseded_by"]
                if self._retired_at(conn, by, row["id"]) <= occurred_jd:
                    continue
                later[row["id"]] = dict(row)
                valid_to = row["valid_to"]
                if row["superseded_by"]:
                    start = conn.execute("SELECT valid_from FROM source_claims WHERE id=?", (by,)).fetchone()
                    if start is not None and start[0] == valid_to:
                        valid_to = None
                conn.execute("UPDATE source_claims SET retracted_by=NULL,superseded_by=NULL,valid_to=? WHERE id=?",
                             (valid_to, row["id"]))
        return later

    @staticmethod
    def _retired_at(conn, by: str, claim_id: str) -> float:
        """When ``by`` retired ``claim_id``, as a Julian day: the source time of the claim that retired it (or of the
        claim it re-extracted), else when the owner's answer was logged; unknown is later than anything."""
        row = conn.execute(f"SELECT {_occurrence()} FROM source_claims c JOIN turn_sources s ON s.turn_id=c.turn_id "
                           "WHERE c.id=?", (by,)).fetchone() or conn.execute(
            f"SELECT {_occurrence()} FROM source_claim_reextractions c JOIN turn_sources s ON s.turn_id=c.turn_id "
            "WHERE c.claim_id=?", (by,)).fetchone()
        if row is not None and row[0] is not None:
            return float(row[0])
        logged = conn.execute("SELECT max(at) FROM learning_log WHERE store='claim' AND item_id=? AND "
                              "action='declined'", (claim_id,)).fetchone()
        if logged is not None and logged[0] is not None:
            return float(logged[0]) / 86400.0 + 2440587.5
        return math.inf

    def _keep_retired(self, conn, contact: str, item: _New, occurred_jd: float, own_ids: set) -> int:
        """A value the message states for the first time, which the store retracted or superseded after the message
        was said, is retracted or superseded with it: the same correction applies to it. A claim a re-extraction
        retired still says what was corrected under its key (its successor may hold another key): it counts, with
        the correction's current holder."""
        key = (contact, item.id, item.subject_key, item.predicate, item.value_key)
        rows = conn.execute('''SELECT c.id,c.retracted_by,c.superseded_by,c.valid_to,0 AS archived FROM source_claims c
                JOIN turn_sources s ON s.turn_id=c.turn_id WHERE s.contact_id=? AND s.scope='person' AND c.id<>?
                AND c.subject_key=? AND c.predicate=? AND c.value_key=?
                AND (c.retracted_by IS NOT NULL OR c.superseded_by IS NOT NULL) ORDER BY c.id''', key).fetchall()
        rows += conn.execute('''SELECT a.claim_id AS id,a.retracted_by,a.superseded_by,a.valid_to,1 AS archived
                FROM source_claim_reextractions a JOIN turn_sources s ON s.turn_id=a.turn_id WHERE s.contact_id=?
                AND s.scope='person' AND a.claim_id<>? AND a.subject_key=? AND a.predicate=? AND a.value_key=?
                AND (a.retracted_by IS NOT NULL OR a.superseded_by IS NOT NULL) ORDER BY a.claim_id''', key).fetchall()
        for row in rows:
            by = row["retracted_by"] or row["superseded_by"]
            if by in own_ids or self._retired_at(conn, by, row["id"]) <= occurred_jd:
                continue
            if row["archived"]:
                by = self._holder(conn, by)
                if by in own_ids or by == item.id:
                    continue
            mine = conn.execute("SELECT retracted_by,superseded_by,valid_from,valid_to FROM source_claims WHERE id=?",
                                (item.id,)).fetchone()
            if mine["retracted_by"] or mine["superseded_by"]:
                return 0
            if row["retracted_by"]:
                conn.execute("UPDATE source_claims SET retracted_by=? WHERE id=?", (by, item.id))
                return 1
            if mine["valid_from"] is None or row["valid_to"] is None or mine["valid_from"] < row["valid_to"]:
                valid_to = min(filter(None, (mine["valid_to"], row["valid_to"])), default=None)
                conn.execute("UPDATE source_claims SET superseded_by=?,valid_to=? WHERE id=?", (by, valid_to, item.id))
                return 1
        return 0

    def _decided(self, conn, pending: Dict[str, Any], old: List[_Old], successor: Dict[str, tuple],
                 pending_before: set) -> int:
        """A held change this commit proposed that the owner already decided (for the earlier claims it re-reads) is
        decided the same way; one the owner is still to decide is not proposed twice."""
        def earlier(ident):
            found = {ident} | {row.id for row in old if successor.get(row.id) and successor[row.id][0].id == ident}
            frontier = list(found)
            while frontier:
                marks = ",".join("?" for _ in frontier)
                frontier = [row[0] for row in conn.execute(
                    f"SELECT claim_id FROM source_claim_reextractions WHERE reextracted_by IN ({marks})", frontier)
                    if row[0] not in found]
                found.update(frontier)
            return found

        targets, bys = earlier(pending["target_id"]), earlier(pending["by_id"])
        t_marks, b_marks = ",".join("?" for _ in targets), ",".join("?" for _ in bys)
        prior = conn.execute(f"SELECT * FROM learning_pending WHERE id<>? AND kind=? AND target_id IN ({t_marks}) "
                             f"AND by_id IN ({b_marks}) ORDER BY (status='pending'),coalesce(decided_at,0) DESC "
                             "LIMIT 1", (pending["id"], pending["kind"], *targets, *bys)).fetchone()
        if prior is None:
            return 0
        if prior["status"] == "pending":
            conn.execute("DELETE FROM learning_pending WHERE id=?", (pending["id"],))
            return 1
        from colonymind.memory import learning
        if prior["status"] == "ratified":
            by = conn.execute("SELECT data_json,valid_from FROM source_claims WHERE id=?",
                              (pending["by_id"],)).fetchone()
            if by is not None:
                claim = json.loads(by["data_json"])
                claim.update(operation=pending["operation"], valid_from=by["valid_from"])
                self.projection._retire(conn, [pending["target_id"]], claim, pending["by_id"])
        conn.execute("UPDATE learning_pending SET status=?,decided_at=?,decided_by=?,reason=? WHERE id=?",
                     (prior["status"], time.time(), prior["decided_by"],
                      ("as decided in " + prior["id"])[:200], pending["id"]))
        learning.log(conn, store="claim", action="ratified" if prior["status"] == "ratified" else "declined",
                     stage="reextract", reason="as the owner decided before", item_id=pending["target_id"],
                     contact_id=pending["contact_id"], refs=[pending["by_id"], prior["id"]])
        return 1

    def _carry_relations(self, pairs: List[tuple]) -> None:
        """The running mind's decided relations (``claims.relation:``) follow a claim to its successor with the same
        value; best effort: the mind may be elsewhere or not running."""
        try:
            state = self.relations() if callable(self.relations) else None
            if state is None:
                return
            from colonymind.mind.consolidate import RELATION_KEY, _pair_id
            mapping = dict(pairs)
            for entry in state.items(RELATION_KEY):
                causes = list(entry.get("causes") or [])
                if len(causes) < 2 or not (causes[0] in mapping or causes[1] in mapping):
                    continue
                first, second = mapping.get(causes[0], causes[0]), mapping.get(causes[1], causes[1])
                state.set(RELATION_KEY + _pair_id(first, second), text=entry.get("text"),
                          causes=[first, second, *causes[2:]])
        except Exception as exc:
            logger.warning("re-extraction left a decided relation behind (%s)", type(exc).__name__)
