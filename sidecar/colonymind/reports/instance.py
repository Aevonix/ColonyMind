"""A running instance's stores, opened for reading only.

The report may run beside a live sidecar, so it never opens a store the way the stores themselves do (their
constructors create tables and run migrations). Each SQLite file is opened read-only: a WAL database that a
process has open (its ``-shm`` index is there) through SQLite's own shared-memory protocol, so the reader sees a
consistent snapshot while the sidecar writes; a WAL database nobody has open (no ``-shm``) as an immutable file,
because a read-only connection would otherwise create the ``-wal`` and ``-shm`` files beside it. SQLite never
checks an immutable file for change, so the file's identity, size and modification time are kept when it is
opened and checked after every read: a writer that opened it and checkpointed since would leave the connection
mixing old pages with new ones, so that read is thrown away and the store opened again. ``query_only`` refuses
any write on every connection. A store that is missing is reported as missing, one that cannot be read as
unreadable; neither stops the report. The source ledger uses a private database/WAL copy, checked against the
original after each read, so its cached tombstone reader cannot change even native shared-memory bytes.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

INITIATIVES, MIND, LEDGER = "initiatives.db", "mind.db", "turn-idempotency.db"
OFF_MARKER = "mind.off"
READS = 3                 # attempts at a read of an immutable file that keeps changing under it


def signature(path: Path) -> Optional[Tuple[int, int, int]]:
    """The file's identity, size and modification time; None when it cannot be seen."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns



@contextmanager
def source_snapshot(path):
    """A private consistent copy avoids even native WAL read-mark writes.

    Report reads must not change the instance's files. A cached live fence
    keeps the native WAL open, where a normal SQLite reader can change -shm.
    Copy only the database and WAL, and let SQLite rebuild the private index.
    """
    import shutil
    import tempfile
    from colonymind.memory import curated
    from types import SimpleNamespace
    directory = Path(tempfile.mkdtemp(prefix='colonymind-work-report-'))
    target = directory / 'snapshot.db'
    paths = [path, Path(str(path) + '-wal')]
    try:
        for _ in range(3):
            before = [signature(item) for item in paths]
            try:
                for item, version in zip(paths, before):
                    copy_path = target if item == path else Path(str(target) + '-wal')
                    if version is not None:
                        shutil.copyfile(item, copy_path)
                    elif copy_path.exists():
                        copy_path.unlink()
            except FileNotFoundError:
                continue
            if before == [signature(item) for item in paths]:
                break
        else:
            raise sqlite3.DatabaseError('work_source_snapshot_changed')
        with closing(sqlite3.connect(target)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA query_only=ON')
            yield conn
            if before != [signature(item) for item in paths]:
                raise sqlite3.DatabaseError('work_source_snapshot_changed')
    finally:
        # These are our private copies only; no instance file is touched and
        # cleanup deliberately does not use a recursive directory remover.
        cache = curated._cache_of(SimpleNamespace(db_path=target)) if target.exists() else None
        if cache is not None:
            with cache.lock:
                cache.monitor.close()
        for suffix in ('-wal', '-shm', '-journal', ''):
            Path(str(target) + suffix).unlink(missing_ok=True)
        directory.rmdir()


def _open(path: Path) -> Tuple[Optional[sqlite3.Connection], Optional[Tuple[int, int, int]]]:
    """``(connection, signature)``: the signature when the file is opened immutable (None otherwise), and no
    connection when it does not exist. Raises ``sqlite3.Error`` or ``OSError`` when it cannot be read."""
    path = Path(path)
    if not path.is_file():
        return None, None
    before = signature(path)
    with path.open("rb") as stream:
        header = stream.read(20)
    wal = len(header) == 20 and header[:16] == b"SQLite format 3\x00" and header[18] == 2
    live = path.with_name(path.name + "-shm").exists()
    immutable = wal and not live
    query = "mode=ro&immutable=1" if immutable else "mode=ro"
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?{query}", uri=True, timeout=5.0, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except BaseException:
        conn.close()
        raise
    return conn, before if immutable else None


def open_readonly(path: Path) -> Optional[sqlite3.Connection]:
    """``path`` read-only, or None when it does not exist. Raises ``sqlite3.Error`` or ``OSError`` when it cannot
    be read."""
    return _open(path)[0]


def parse_time(value: Any) -> Optional[datetime]:
    """An ISO timestamp or a Unix time as an aware UTC datetime; None when it is neither."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        with suppress(OverflowError, OSError, ValueError):
            return datetime.fromtimestamp(float(value), timezone.utc)
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(text), timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def loads(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(value) if value else default
    except (TypeError, ValueError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


class Instance:
    """The instance directory: its configuration and its stores, read-only."""

    def __init__(self, home: Any) -> None:
        from colonymind.config import load_config
        self.home = Path(home).expanduser()
        # The file's settings as the sidecar reads them; the report's own environment never overrides them.
        self.config = load_config(self.home, environ={})
        self.missing: List[str] = []
        self.unreadable: List[str] = []
        self.skipped = 0      # intention rows the report could not parse (an older or hand-edited shape)
        self._conns: Dict[str, Optional[sqlite3.Connection]] = {}
        self._frozen: Dict[str, Tuple[int, int, int]] = {}   # the signature each immutable store was opened at
        self._copies = {}
        self._versions = {}

    def _version(self, name):
        path = self.home / name
        return signature(path), signature(Path(str(path) + '-wal'))

    def _open_source(self, name):
        path = self.home / name
        if not path.is_file():
            return None, None
        for _ in range(READS):
            before = self._version(name)
            context = source_snapshot(path)
            conn = context.__enter__()
            if before == self._version(name):
                self._copies[name], self._versions[name] = context, before
                return conn, before[0]
            with suppress(sqlite3.DatabaseError):
                context.__exit__(None, None, None)
        raise sqlite3.DatabaseError('report_source_snapshot_changed')

    def db(self, name: str) -> Optional[sqlite3.Connection]:
        if name not in self._conns:
            try:
                conn, frozen = self._open_source(name) if name == LEDGER else _open(self.home / name)
            except (sqlite3.Error, OSError):     # a damaged store, or one this user may not open
                conn, frozen = None, None
                self._unreadable(name)
            else:
                if conn is None and name not in self.missing:
                    self.missing.append(name)
            self._conns[name] = conn
            if frozen is not None:
                self._frozen[name] = frozen
        return self._conns[name]

    def _unreadable(self, name: str) -> None:
        if name not in self.unreadable:
            self.unreadable.append(name)

    def _reopen(self, name: str) -> None:
        conn = self._conns.pop(name, None)
        self._frozen.pop(name, None)
        context = self._copies.pop(name, None)
        self._versions.pop(name, None)
        if context is not None:
            with suppress(sqlite3.DatabaseError):
                context.__exit__(None, None, None)
        elif conn is not None:
            conn.close()

    def _unchanged(self, name):
        if name in self._versions:
            return self._versions[name] == self._version(name)
        return name not in self._frozen or self._frozen[name] == signature(self.home / name)

    def query(self, name: str, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Rows of ``sql`` on store ``name``; none when the store or the table is absent or unreadable."""
        return self.read(name, lambda conn: conn.execute(sql, params).fetchall(), [])

    def read(self, name: str, reader: Callable[[sqlite3.Connection], Any], default: Any = None) -> Any:
        """``reader(connection)`` on store ``name`` (read-only, as ``query`` reads), for a read the running code
        makes itself; ``default`` when the store or a table it reads is absent or unreadable."""
        for _ in range(READS):
            conn = self.db(name)
            if conn is None:
                return default
            done = False
            try:
                result, done = reader(conn), True
            except sqlite3.OperationalError as error:
                if self._unchanged(name):
                    if "no such table" not in str(error) and "no such column" not in str(error):
                        self._unreadable(name)
                    return default
            except sqlite3.DatabaseError:
                if self._unchanged(name):
                    self._unreadable(name)
                    return default
            if done and self._unchanged(name):
                return result
            self._reopen(name)      # the immutable file changed since it was opened: read it again
        self._unreadable(name)
        return default

    def close(self) -> None:
        for name in list(self._conns):
            self._reopen(name)

    @property
    def owner_id(self) -> str:
        return str(self.config.get("owner.contact_id") or "")

    @property
    def mind_enabled(self) -> bool:
        return self.config.get("mind.enabled") is not False and not (self.home / OFF_MARKER).exists()

    @property
    def autonomy(self) -> str:
        return str(self.config.get("mind.autonomy") or "suggest")

    def setting(self, dotted: str) -> bool:
        """A faculty's switch as the file sets it (the shipped default where it is silent). A mind faculty that
        neither the file nor the shipped defaults name is on (``config.MEANING_FACULTIES``)."""
        value = self.config.get(dotted)
        return value is True or (value is None and dotted.startswith("mind.faculties."))

    def mind_rows(self, start: datetime, end: datetime) -> List[Any]:
        """The mind's intention rows formed in ``[start, end)``, oldest first."""
        from colonymind.initiatives.models import StoredInitiative
        margin = (start - timedelta(days=1)).astimezone(timezone.utc).isoformat()
        rows = []
        for row in self.query(INITIATIVES, "SELECT * FROM initiatives WHERE kind IS NOT NULL AND created_at >= ? "
                                           "ORDER BY created_at", (margin,)):
            at = parse_time(row["created_at"])
            if at is None:
                self.skipped += 1       # formed at no time it can read: in no window
                continue
            if not start <= at < end:
                continue
            try:
                rows.append(StoredInitiative.from_row(dict(row)))
            except (TypeError, ValueError, KeyError):
                self.skipped += 1
        return rows

    def open_rows(self, statuses: tuple) -> List[Any]:
        """Mind rows still in one of ``statuses``, whenever they formed."""
        from colonymind.initiatives.models import StoredInitiative
        marks = ",".join("?" * len(statuses))
        rows = []
        for row in self.query(INITIATIVES, f"SELECT * FROM initiatives WHERE kind IS NOT NULL AND status IN ({marks})",
                              statuses):
            try:
                rows.append(StoredInitiative.from_row(dict(row)))
            except (TypeError, ValueError, KeyError):
                self.skipped += 1
        return rows

    def handed_over(self, ids: List[str]) -> Dict[str, datetime]:
        """When each of ``ids`` was last handed to the body: its latest approval (``queued``) or dispatch
        (``bound``) in the audit; an id with neither is left out."""
        found: Dict[str, datetime] = {}
        for index in range(0, len(ids), 500):
            chunk = ids[index:index + 500]
            marks = ",".join("?" * len(chunk))
            for row in self.query(INITIATIVES, f"SELECT initiative_id, timestamp FROM assignment_history WHERE "
                                               f"action IN ('queued', 'bound') AND initiative_id IN ({marks})",
                                  tuple(chunk)):
                at = parse_time(row["timestamp"])
                if at is not None and (row["initiative_id"] not in found or at > found[row["initiative_id"]]):
                    found[row["initiative_id"]] = at
        return found

    def history(self, actions: tuple, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """The audit transitions named ``actions`` in the window: ``{id, action, at, details}``."""
        marks = ",".join("?" * len(actions))
        margin = (start - timedelta(days=1)).astimezone(timezone.utc).isoformat()
        found = []
        for row in self.query(INITIATIVES, f"SELECT initiative_id, action, timestamp, details FROM assignment_history "
                                           f"WHERE action IN ({marks}) AND timestamp >= ?", (*actions, margin)):
            at = parse_time(row["timestamp"])
            if at is not None and start <= at < end:
                found.append({"id": row["initiative_id"], "action": row["action"], "at": at,
                              "details": loads(row["details"], {})})
        return found


__all__ = ["INITIATIVES", "Instance", "LEDGER", "MIND", "loads", "open_readonly", "parse_time"]
