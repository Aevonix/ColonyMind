"""Model usage by lane: one row of counts per model call, in the instance's ``usage.db`` (docs/FUNCTION-ROUTING.md,
Usage).

Every model client the sidecar runs records its calls: the router (each attempt, so a fallback's failed attempt is a
row of its own), Instinct's decision providers (a row per question asked, the reader's per completions call), and
the embedder and reranker. A row holds when the call ended, the client, the caller's task, the role, the instance's
own name for the endpoint, input and output tokens (each none when the endpoint did not report it, never a zero; an
embedding or a rerank generates no tokens, so its output count is 0), latency and the outcome.
Never the words of a prompt or an answer, a URL, a model's name, or an id of a person or a source: a task is kept
only as a code name (a lowercase letter, then lowercase letters and ``_``, so no digit or id in a caller's text can
pass) or as a name the process registered (``register_tasks``: the decision points, whose names may hold digits,
as an operator's ``cron_job_1``); anything else is ``unnamed``. A role or an endpoint is kept as it is only when it
is a plain name (a letter, then letters, digits, ``_`` or ``-``). Any other name (``worker.v1``, a host name, an
address) is kept as ``alias-`` and the first 12 hex digits of an HMAC-SHA256 under the instance's own secret
(``colonymind.instance.instance_secret``, ``instinct.key``): two names never merge, none is kept as text, and since an
address is a small space to search, the key keeps the ledger alone from testing guesses (the same name has another
alias in every instance; ``alias_of`` and ``colonymind report usage --alias NAME`` say which it has here, and
``earlier_alias_of`` the one an earlier build's rows still hold until a write keys them). The writer
makes the alias, so a call never reads the secret; a row whose alias cannot be keyed is dropped, never kept unkeyed.
Only a missing name is ``other``.

The key is taken over the name's SHA-256 prefix (``digest_alias``), the form an earlier build kept as the alias
itself. So a ledger that build wrote is keyed in place: the writer's first transaction keys each alias it holds
once and records that it did (``PRAGMA user_version``, ``KEYED``), and a name's earlier rows and its new ones stay
one group, none left with an alias a guess could test.

An embedder cannot tell from its own call what an embedding is for, so the caller names it (``embedding_task``): the
text of a captured source is ``source_embedding`` (a memory lane), re-embedding it for a new index ``source_reindex``,
a warmup or a health check ``embedding_check``; a live query is ``query_embedding`` and any other text
``text_embedding``, the embedder's own defaults where no caller named one.

The same file counts captures (``captured``): for each newly captured person-scoped source, when it was captured and
how many user messages it held, and nothing else. The memory lanes' cost per captured message divides by these
counts, which erasing a source never changes: its calls were made, whatever became of its words.

A call hands its row to a queue and goes on: one writer thread appends the rows, so the ledger costs a model call
no disk time, and a ledger that cannot be written costs it nothing (the row is dropped and counted), nor does a
writer thread that cannot start (the rows wait in the queue for the next call that starts one). Rows go only
to a state directory the process names (``COLONYMIND_STATE_DIR``, else ``COLONYMIND_HOME``, as the sidecar's
configuration sets them) and that exists: a harness or a script that names none records nothing, never into a
default instance. The sidecar's daily maintenance (``instinct.maintenance``) drops rows older than ``RETENTION``
(``prune``), whether or not the instance still makes calls. ``colonymind report usage`` reads the table
(``reports/usage.py``). A missing ledger, or one whose tables are not written yet, reads as no usage; one that
cannot be read (locked past ``READ_WAIT``, or damaged) raises ``Unreadable``, never reads as zero.
"""
from __future__ import annotations

import asyncio
import atexit
import contextvars
import functools
import hashlib
import hmac
import logging
import os
import queue
import re
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

DB_FILE = "usage.db"
#: How long a row is kept (a chosen value): a month of usage for the report and the cost estimates.
RETENTION = timedelta(days=30)
#: Rows waiting for the writer before new ones are dropped (my choice): a stalled disk never grows the process.
MAX_PENDING = 10_000
#: Rows the writer appends in one transaction at most.
BATCH = 500
#: Seconds a reader waits for the writer's lock before the ledger is ``Unreadable`` (SQLite's busy wait).
READ_WAIT = 5.0
CLIENTS = ("router", "instinct", "embedder", "reranker")
#: The clients whose calls generate no tokens: an embedding's or a rerank's output count is 0, not unknown.
NO_OUTPUT = ("embedder", "reranker")
OUTCOMES = ("ok", "error", "timeout", "incomplete", "cancelled")
COLUMNS = ("at", "client", "task", "role", "endpoint", "input_tokens", "output_tokens", "latency_ms", "outcome")
SCHEMA = """
CREATE TABLE IF NOT EXISTS model_calls (
  at TEXT NOT NULL,
  client TEXT NOT NULL,
  task TEXT NOT NULL,
  role TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  input_tokens INTEGER,
  output_tokens INTEGER,
  latency_ms INTEGER NOT NULL,
  outcome TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_calls_at ON model_calls(at);
CREATE TABLE IF NOT EXISTS captures (
  at TEXT NOT NULL,
  messages INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_captures_at ON captures(at);
"""
_INSERT = f"INSERT INTO model_calls ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})"
_INSERT_CAPTURE = "INSERT INTO captures (at, messages) VALUES (?, ?)"
_TASK = re.compile(r"[a-z][a-z_]{0,63}")
#: A registered name's shape (a decision point's: lowercase letters, digits and ``_``).
_REGISTERED_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_REGISTERED: frozenset = frozenset()
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
#: Hex digits of a name's keyed hash in its alias (my choice: 48 bits, far past a collision among an instance's
#: names).
ALIAS_DIGITS = 12
#: The ledger's ``user_version`` once every alias in it is keyed (``_key_aliases``); 0: written by an earlier build,
#: whose aliases are the plain digest prefix (``digest_alias``).
KEYED = 1
_DIGEST_ALIAS = re.compile(r"alias-[0-9a-f]{%d}" % ALIAS_DIGITS)
_MAX_COUNT = 10 ** 12
#: What the embeddings made in the current block are for (``embedding_task``); unset, the embedder's own default.
_EMBEDDING_TASK: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("colonymind_embedding_task",
                                                                                  default=None)


class Unreadable(RuntimeError):
    """The ledger exists and could not be read: locked past ``READ_WAIT``, or damaged. Its usage is unknown, never
    zero. ``reason`` is SQLite's own words (``database is locked``), which hold nothing from the rows."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"the model usage ledger could not be read ({reason})")
        self.reason = reason


def register_tasks(names: Iterable[Any]) -> None:
    """Record these names as they are when a call gives one as its task: the decision points of a registry, as the
    process builds it (the mind's points, and each engine's). A name outside the point shape is never registered."""
    global _REGISTERED
    _REGISTERED = _REGISTERED | {value for value in names
                                 if isinstance(value, str) and _REGISTERED_NAME.fullmatch(value)}


def task_name(value: Any) -> str:
    """A caller's task as it is recorded: a code name or a registered name, or ``unnamed``."""
    if isinstance(value, str) and (_TASK.fullmatch(value) or value in _REGISTERED):
        return value
    return "unnamed"


@contextmanager
def embedding_task(task: str) -> Iterator[None]:
    """Name what the embeddings made inside this block are for, as their rows' task: an embedder's call cannot tell a
    captured source's text from a health check. The innermost block's name holds; the embedder's own default
    (``embedding_task_of``) applies only where no caller named one."""
    token = _EMBEDDING_TASK.set(task)
    try:
        yield
    finally:
        _EMBEDDING_TASK.reset(token)


def embedding_task_of(default: str) -> str:
    """The task the caller named for this embedding (``embedding_task``), else ``default``."""
    return _EMBEDDING_TASK.get() or default


def plain(value: Any) -> Optional[str]:
    """What the table keeps for a role or an endpoint without an alias: a plain name as it is, ``other`` when there
    is none; None for a name that is kept as its alias."""
    if not isinstance(value, str) or not value:
        return "other"
    return value if _NAME.fullmatch(value) else None


def name(value: Any, secret: bytes) -> str:
    """A role or an endpoint's name as it is recorded: a plain name as it is, any other name as its alias under the
    instance's ``secret`` (``alias``), and ``other`` when there is none."""
    kept = plain(value)
    return kept if kept is not None else alias(value, secret)


def digest_alias(value: str) -> str:
    """``alias-`` and the first ``ALIAS_DIGITS`` hex digits of the name's plain SHA-256: what the keyed alias is
    taken over, and the alias itself in a ledger an earlier build wrote (``_key_aliases`` keys it in place)."""
    return "alias-" + hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:ALIAS_DIGITS]


def _keyed(digested: str, secret: bytes) -> str:
    return "alias-" + hmac.new(secret, digested.encode("ascii"), hashlib.sha256).hexdigest()[:ALIAS_DIGITS]


@functools.lru_cache(maxsize=256)
def alias(value: str, secret: bytes) -> str:
    """A name that is not plain (a dot, a leading digit, a space) as the table keeps it: ``alias-`` and the first
    ``ALIAS_DIGITS`` hex digits of an HMAC-SHA256 under the instance's ``secret`` of its digest prefix
    (``digest_alias``), distinct for each name and never its text."""
    return _keyed(digest_alias(value), secret)


def alias_of(directory: Path, value: str) -> Optional[str]:
    """What the rows of the instance at ``directory`` record for the name ``value``: the name itself when it is
    plain, else its alias there. None when the instance has no secret yet (no alias was recorded); never makes
    one."""
    kept = plain(value)
    if kept is not None:
        return kept
    from colonymind.instance import instance_secret
    secret = instance_secret(Path(directory).expanduser(), create=False)
    return None if secret is None else alias(value, secret)


def earlier_alias_of(directory: Path, value: str) -> Optional[str]:
    """The alias rows an earlier build wrote still hold for the name ``value`` in ``directory``'s ledger: its plain
    digest prefix (``digest_alias``), while no write of this build has keyed the ledger yet (``KEYED``) and a row
    holds it. None otherwise, and for a plain name. Read-only; a ledger that cannot be read raises ``Unreadable``."""
    if plain(value) is not None:
        return None
    earlier = digest_alias(value)
    with _reading(directory, "model_calls") as conn:
        if conn is None or conn.execute("PRAGMA user_version").fetchone()[0] >= KEYED:
            return None
        held = conn.execute("SELECT 1 FROM model_calls WHERE role = ? OR endpoint = ? LIMIT 1",
                            (earlier, earlier)).fetchone()
    return earlier if held else None


def _count(value: Any) -> Optional[int]:
    return value if type(value) is int and 0 <= value < _MAX_COUNT else None


def stamp(moment: datetime) -> str:
    """A time as the table holds it: UTC, to the second."""
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def outcome_of(error: Optional[BaseException]) -> str:
    """How a call ended: ``ok``; ``cancelled`` (the caller stopped waiting, as a decision tier does at its deadline);
    ``timeout`` (the client's own time limit); ``incomplete`` (an answer came that could not be used); ``error``
    (anything else: the endpoint refused or could not be reached)."""
    if error is None:
        return "ok"
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return "timeout"
    try:
        import httpx
        if isinstance(error, httpx.HTTPError):
            return "error"
    except ImportError:                                        # pragma: no cover - httpx ships with the sidecar
        pass
    if isinstance(error, OSError):
        return "error"
    if isinstance(error, (ValueError, LookupError)):     # an answer without a field, or an index outside it
        return "incomplete"
    return "error"


def reported_tokens(report: Any) -> Tuple[Optional[int], Optional[int]]:
    """Input and output tokens from an endpoint's own usage report (an OpenAI-style ``usage`` mapping or object):
    ``prompt_tokens`` (or ``input_tokens``; ``total_tokens`` when it reports nothing else, as embeddings and
    reranks do) and ``completion_tokens`` (or ``output_tokens``). None for a count it did not report."""
    if report is None:
        return None, None
    read = report.get if isinstance(report, Mapping) else (lambda key: getattr(report, key, None))
    output = next((value for value in (_count(read("completion_tokens")), _count(read("output_tokens")))
                   if value is not None), None)
    given = next((value for value in (_count(read("prompt_tokens")), _count(read("input_tokens")))
                  if value is not None), None)
    if given is None and output is None:
        given = _count(read("total_tokens"))
    return given, output


def function_usage(outcome: Any) -> Dict[str, int]:
    """What one function call used by its endpoints' own reports, summed over every attempt: a router response's
    ``usage`` plus its ``prior_usage`` (the attempts that failed before it), or the ``usage`` the router puts on the
    error it raises when no attempt completed (a completion that ran out while thinking still spent its whole cap).
    Counts only (``prompt_tokens``, ``completion_tokens``, ``total_tokens``); empty when nothing was reported, as
    for an error the router never reached an endpoint for."""
    total: Dict[str, int] = {}
    for report in (getattr(outcome, "usage", None), getattr(outcome, "prior_usage", None)):
        if isinstance(report, Mapping):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                count = report.get(key)
                if type(count) is int and count > 0:
                    total[key] = total.get(key, 0) + count
    return total


def charged_tokens(outcome: Any) -> int:
    """The tokens to charge a budget for one function call (``function_usage``): the reported total, else prompt plus
    completion; 0 when nothing was reported."""
    used = function_usage(outcome)
    return used.get("total_tokens") or used.get("prompt_tokens", 0) + used.get("completion_tokens", 0)


def _named_directory() -> Optional[str]:
    return os.environ.get("COLONYMIND_STATE_DIR") or os.environ.get("COLONYMIND_HOME") or None


def record(*, client: str, task: Any, role: Any, endpoint: Any, latency_ms: Any, outcome: str,
           input_tokens: Any = None, output_tokens: Any = None) -> None:
    """Hand one call's row to the writer. Never raises and never waits: a row that cannot be queued is dropped."""
    try:
        directory = _named_directory()
        if directory is None:
            return
        latency = int(latency_ms) if isinstance(latency_ms, (int, float)) and latency_ms == latency_ms else 0
        # A name that needs an alias stays in the queue (this process's memory) until the writer keys it.
        row = (stamp(now_utc()), client if client in CLIENTS else "other", task_name(task),
               role if isinstance(role, str) else None, endpoint if isinstance(endpoint, str) else None,
               _count(input_tokens), _count(output_tokens), min(max(0, latency), _MAX_COUNT),
               outcome if outcome in OUTCOMES else "error")
    except Exception:
        _WRITER.dropped += 1
        return
    _WRITER.put(directory, row)


def captured(messages: Any) -> None:
    """Count one newly captured person-scoped source's user messages: a time and a number, never which source or
    whose. Never raises and never waits, like ``record``."""
    try:
        directory = _named_directory()
        count = _count(messages)
        if directory is None or not count:
            return
        row = (stamp(now_utc()), count)
    except Exception:
        _WRITER.dropped += 1
        return
    _WRITER.put(directory, row, _INSERT_CAPTURE)


class Call:
    """Times one model call and records it when the block ends, however it ends (the error is never swallowed)."""

    __slots__ = ("client", "task", "role", "endpoint", "input_tokens", "output_tokens", "_started")

    def __init__(self, client: str, *, task: Any, role: Any, endpoint: Any) -> None:
        self.client, self.task, self.role, self.endpoint = client, task, role, endpoint
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = 0 if client in NO_OUTPUT else None
        self._started = 0.0

    def tokens(self, report: Any) -> None:
        """Take the counts from the endpoint's usage report (``reported_tokens``); an embedding's or a rerank's
        output count stays 0 when the report has none."""
        given, output = reported_tokens(report)
        self.input_tokens = given
        self.output_tokens = output if output is not None or self.client not in NO_OUTPUT else 0

    def __enter__(self) -> "Call":
        self._started = time.monotonic()
        return self

    def __exit__(self, kind, error, trace) -> bool:
        record(client=self.client, task=self.task, role=self.role, endpoint=self.endpoint,
               latency_ms=(time.monotonic() - self._started) * 1000.0, outcome=outcome_of(error),
               input_tokens=self.input_tokens, output_tokens=self.output_tokens)
        return False


def call(client: str, *, task: Any, role: Any, endpoint: Any) -> Call:
    """``with usage.call("reranker", task="rerank", role="rerank", endpoint="rerank") as call: ...``"""
    return Call(client, task=task, role=role, endpoint=endpoint)


class _Writer:
    """The one thread that appends rows, in the order they were queued."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Tuple[Optional[str], str, Any]]" = queue.Queue(maxsize=MAX_PENDING)
        self._thread: Optional[threading.Thread] = None
        self._start = threading.Lock()
        self.dropped = 0

    def put(self, directory: str, row: tuple, insert: str = _INSERT) -> None:
        try:
            self._queue.put_nowait((directory, insert, row))
        except Exception:                   # full, or a queue that cannot take it: the call goes on without it
            self.dropped += 1
            return
        self._ensure()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every row queued before this call is written (or dropped); False when that took too long or
        no writer could start."""
        if self._thread is None and self._queue.empty():
            return True
        done = threading.Event()
        try:
            self._queue.put((None, "", done), timeout=timeout)
        except Exception:
            return False
        return self._ensure() and done.wait(timeout)

    def _ensure(self) -> bool:
        """A running writer; False when none could start (a process at its thread limit). Never raises: the rows
        wait in the queue for the next call that starts one, and a full queue drops and counts new rows."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            return True
        try:
            with self._start:
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(target=self._run, name="colonymind-usage", daemon=True)
                    self._thread.start()
            return True
        except Exception as error:
            logger.debug("model usage writer not started (%s)", type(error).__name__)
            return False

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < BATCH:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._write(batch)

    def _write(self, batch: List[Tuple[Optional[str], str, Any]]) -> None:
        rows: Dict[str, Dict[str, List[tuple]]] = {}
        flushes = []
        for directory, insert, item in batch:
            if directory is None:
                flushes.append(item)
            else:
                rows.setdefault(directory, {}).setdefault(insert, []).append(item)
        for directory, inserts in rows.items():
            try:
                self._append(Path(directory), inserts)
            except Exception as error:
                self.dropped += sum(len(items) for items in inserts.values())
                logger.debug("model usage rows dropped (%s)", type(error).__name__)
        for done in flushes:
            done.set()

    def _named(self, directory: Path, items: List[tuple]) -> List[tuple]:
        """The call rows with their role and endpoint as the table keeps them: a plain name as it is, any other as
        its alias under the instance's secret, read (or made) once for the batch and only when a name needs it. A row
        whose alias cannot be keyed (a damaged secret file) is dropped and counted, never kept unkeyed."""
        from colonymind.instance import instance_secret
        named, secret, failed = [], None, False
        for row in items:
            role, endpoint = plain(row[3]), plain(row[4])
            if role is None or endpoint is None:
                if secret is None and not failed:
                    try:
                        secret = instance_secret(directory)
                    except Exception as error:
                        failed = True
                        logger.debug("model usage aliases not keyed (%s)", type(error).__name__)
                if secret is None:
                    self.dropped += 1
                    continue
                role, endpoint = name(row[3], secret), name(row[4], secret)
            named.append((*row[:3], role, endpoint, *row[5:]))
        return named

    def _append(self, directory: Path, inserts: Dict[str, List[tuple]]) -> None:
        if not directory.is_dir():
            self.dropped += sum(len(items) for items in inserts.values())
            return
        if _INSERT in inserts:
            inserts = {**inserts, _INSERT: self._named(directory, inserts[_INSERT])}
        path = directory / DB_FILE
        with closing(sqlite3.connect(path, timeout=5)) as conn:
            conn.executescript(SCHEMA)
            with conn:
                conn.execute("BEGIN IMMEDIATE")      # one writer keys an earlier build's aliases, and only once
                _key_aliases(conn, directory)
                for insert, items in inserts.items():
                    conn.executemany(insert, items)

    def after_fork(self) -> None:
        """A forked child starts with an empty queue and no writer (the parent's thread did not come along)."""
        self.__init__()


_WRITER = _Writer()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_WRITER.after_fork)


@atexit.register
def _at_exit() -> None:
    _WRITER.flush(timeout=2.0)


def flush(timeout: float = 5.0) -> bool:
    """Wait until the rows queued so far are written; for tests and short-lived commands."""
    return _WRITER.flush(timeout)


def dropped() -> int:
    """Rows this process could not record (queue full, ledger unwritable, no such directory)."""
    return _WRITER.dropped


def _key_aliases(conn: sqlite3.Connection, directory: Path) -> None:
    """In the writer's transaction: a ledger an earlier build wrote (``user_version`` below ``KEYED``) has its aliases
    (the plain digest prefix, ``digest_alias``) keyed in place under the instance's secret, the form this build writes
    (``alias``), and is marked keyed. Without a readable secret nothing changes and the next write tries again."""
    if conn.execute("PRAGMA user_version").fetchone()[0] >= KEYED:
        return
    found = {value for column in ("role", "endpoint")
             for (value,) in conn.execute(f"SELECT DISTINCT {column} FROM model_calls WHERE {column} LIKE 'alias-%'")
             if _DIGEST_ALIAS.fullmatch(value)}
    if found:
        from colonymind.instance import instance_secret
        try:
            secret = instance_secret(directory)
        except Exception as error:
            logger.debug("model usage aliases of an earlier build not keyed yet (%s)", type(error).__name__)
            return
        for value in found:
            keyed = _keyed(value, secret)
            for column in ("role", "endpoint"):
                conn.execute(f"UPDATE model_calls SET {column} = ? WHERE {column} = ?", (keyed, value))
    conn.execute(f"PRAGMA user_version = {KEYED}")


def _drop_old(conn: sqlite3.Connection, now: datetime) -> int:
    cut = stamp(now - RETENTION)
    conn.execute("DELETE FROM captures WHERE at < ?", (cut,))
    return conn.execute("DELETE FROM model_calls WHERE at < ?", (cut,)).rowcount


def prune(directory: Path, *, now: Optional[datetime] = None) -> int:
    """Drop the rows and capture counts older than ``RETENTION`` (the sidecar's daily maintenance); the number of
    model call rows dropped. A missing ledger is left missing."""
    path = Path(directory) / DB_FILE
    if not path.is_file():
        return 0
    with closing(sqlite3.connect(path, timeout=5)) as conn, conn:
        return _drop_old(conn, now or now_utc())


@contextmanager
def _reading(directory: Path, table: str) -> Iterator[Optional[sqlite3.Connection]]:
    """A read-only connection to ``directory``'s ledger, or None when there is nothing to read: no ledger (never
    created here), or no ``table`` in it yet. Any failure to read it raises ``Unreadable``: a locked or damaged
    ledger is never read as an empty one."""
    path = Path(directory).expanduser().resolve() / DB_FILE
    if not path.is_file():
        yield None
        return
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=READ_WAIT)) as conn:
            found = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
            yield conn if found else None
    except sqlite3.Error as error:
        raise Unreadable(str(error) or type(error).__name__) from error


def rows(directory: Path, *, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """The rows in ``directory``'s ledger (from ``since``), oldest first. Read-only: a missing ledger is empty and
    is never created; one that cannot be read raises ``Unreadable``."""
    query, args = f"SELECT {', '.join(COLUMNS)} FROM model_calls", []
    if since is not None:
        query += " WHERE at >= ?"
        args.append(stamp(since))
    with _reading(directory, "model_calls") as conn:
        if conn is None:
            return []
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query + " ORDER BY rowid", args)]


def captured_since(directory: Path, since: datetime, until: Optional[datetime] = None) -> int:
    """The user messages captured from ``since`` (to ``until``), as counted when they were captured. Read-only; a
    ledger that cannot be read raises ``Unreadable``."""
    query, args = "SELECT COALESCE(SUM(messages), 0) FROM captures WHERE at >= ?", [stamp(since)]
    if until is not None:
        query += " AND at <= ?"
        args.append(stamp(until))
    with _reading(directory, "captures") as conn:
        return 0 if conn is None else int(conn.execute(query, args).fetchone()[0] or 0)


def totals(found: List[Mapping[str, Any]]) -> Dict[str, int]:
    """Calls, failed calls (any outcome but ``ok``), tokens in and out, and for each the calls that reported no
    count: a call can report one and not the other, and the one it did not report is unknown, never a zero."""
    return {"calls": len(found),
            "failed": sum(1 for row in found if row["outcome"] != "ok"),
            "input_tokens": sum(row["input_tokens"] or 0 for row in found),
            "output_tokens": sum(row["output_tokens"] or 0 for row in found),
            "without_input_tokens": sum(1 for row in found if row["input_tokens"] is None),
            "without_output_tokens": sum(1 for row in found if row["output_tokens"] is None)}


def last_day(directory: Path, *, now: Optional[datetime] = None) -> Dict[str, int]:
    """The last 24 hours' totals (``totals``' shape), summed by the database: the health route reads them on every
    call, so no row is loaded. Read-only; a ledger that cannot be read raises ``Unreadable``."""
    empty = {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "without_input_tokens": 0,
             "without_output_tokens": 0}
    with _reading(directory, "model_calls") as conn:
        if conn is None:
            return empty
        found = conn.execute(
            "SELECT COUNT(*), TOTAL(outcome != 'ok'), TOTAL(input_tokens), TOTAL(output_tokens), "
            "TOTAL(input_tokens IS NULL), TOTAL(output_tokens IS NULL) FROM model_calls WHERE at >= ?",
            (stamp((now or now_utc()) - timedelta(days=1)),)).fetchone()
    return dict(zip(empty, (int(value or 0) for value in found)))


def tokens_in_words(counted: Mapping[str, int]) -> str:
    """Tokens in words, never a count the endpoints did not report, each side on its own: ``1,960 tokens in and at
    least 288 out (3 calls reported no output count)``, ``an unknown number`` for a side no call reported, and ``no
    token count reported`` when no call reported either."""
    calls, missing_in, missing_out = counted["calls"], counted["without_input_tokens"], counted["without_output_tokens"]
    if calls and missing_in == calls and missing_out == calls:
        return "no token count reported"

    def side(total: int, missing: int) -> str:
        if calls and missing == calls:
            return "an unknown number"
        return f"at least {total:,}" if missing else f"{total:,}"
    text = f"{side(counted['input_tokens'], missing_in)} tokens in and {side(counted['output_tokens'], missing_out)} out"
    notes = [(missing, word) for missing, word in ((missing_in, "input"), (missing_out, "output")) if missing]
    if notes:
        (count, word), rest = notes[0], notes[1:]
        text += f" ({count:,} {'call' if count == 1 else 'calls'} reported no {word} count"
        text += "".join(f", {count:,} no {word} count" for count, word in rest) + ")"
    return text


def describe(counted: Mapping[str, int]) -> str:
    """Totals in words: ``8 (7 ok), 1,960 tokens in and at least 288 out (2 calls reported no output count)``. A
    call that did not end ``ok`` is no fault of the instance (a fallback answered, a tier moved on), so the words never read as one
    to a reader of the health notes."""
    if not counted["calls"]:
        return "none recorded"
    return f"{counted['calls']:,} ({counted['calls'] - counted['failed']:,} ok), {tokens_in_words(counted)}"


def health_line(directory: Path, *, now: Optional[datetime] = None) -> str:
    """The health line: the last 24 hours' totals in one sentence (``Unreadable`` when the ledger cannot be read)."""
    return f"Model calls in the last 24 hours: {describe(last_day(directory, now=now))}."


__all__ = ["ALIAS_DIGITS", "BATCH", "CLIENTS", "COLUMNS", "Call", "DB_FILE", "KEYED", "MAX_PENDING", "NO_OUTPUT",
           "OUTCOMES", "READ_WAIT", "RETENTION", "SCHEMA", "Unreadable", "alias", "alias_of", "call", "captured",
           "captured_since", "charged_tokens", "describe", "digest_alias", "dropped", "earlier_alias_of", "embedding_task", "embedding_task_of", "flush", "function_usage", "health_line", "last_day", "name",
           "outcome_of", "plain", "prune", "record", "register_tasks", "reported_tokens", "rows", "stamp", "task_name",
           "tokens_in_words", "totals"]
