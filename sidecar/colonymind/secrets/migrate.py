"""Sealing what was stored before capture sealed, and releasing a false positive (docs/SECRETS.md).

``migrate`` runs once (``colonymind secrets migrate``, or at the first start with ``secrets.migrate.on_start``,
guarded by a marker in ``secrets-vault.db``); ``--dry-run`` counts and writes nothing. Per turn, in one
transaction, idempotent and resumable:

1. ``turn_sources``: each message is sealed, the turn's digests are recomputed over the sealed messages, its
   lexical index rows are rewritten and its vector job is queued again. Each changed identity (a message hash,
   the turn's version, its content digest) is recorded as an old to new pair in the same transaction.
2. Lineage: every text column of every store that cites an old identity (claims, annotations, appraisals,
   judgments, facts, the communications log, a child turn's source references) is re-pointed to the new one,
   to a fixpoint (a child turn whose references changed gets a new version too).
3. Every other text column of every store is sealed in place: queue rows, the ingestion responses, the
   communications log, the mind's rows, the other stores (the calibration's column walker).
   Owner-goal tables are excluded: their explicit store hook seals original field bytes while preserving
   immutable audit and receipt hash preimages, and must finish before the completion marker is written.
4. A report of counts only (``secrets-migration-report.json``): what was sealed per store and rule, the
   nightly backups that predate the migration, the legacy secrets backend that was configured, and a read-only
   count of Hermes transcript messages that still hold values (removing those is the owner's call).

Sealing here is strict: when the vault cannot take a value (a locked keyring, an unreadable key) the migration
stops before it changes that turn or row, the stored words stay as they were, the marker is not written, and the
next run resumes. A JSON value is decoded first and its string leaves sealed, each read with its field name too
(``"password": "..."`` names a value the way the serialized text did), so a value is stored as its bytes (a key
block's line breaks, not the escapes that stood for them).

``release`` runs the same rewrite the other way for one reference: its value goes back where it stood.
Nothing here prints or logs a value.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from colonymind.util.temporal import ledger_stamp

from .config import SecretsSettings, current_settings
from .detect import find_secrets
from .seal import Scope, map_leaves, map_message, redact_sensitive_messages, seal_text, unseal_text
from .vault import Vault, VaultUnavailable, get_vault

logger = logging.getLogger(__name__)

# Earlier markers cannot certify the physical owner-store scrub.
MARKER = "migration:v3-owner-goals-physical"
REPORT_NAME = "secrets-migration-report.json"
LEDGER_NAME = "turn-idempotency.db"
PENDING_TABLE = "secret_lineage_pending"
# The stores that keep what was said, or what was derived from it (the stores the calibration read). Only these
# are rewritten: a store of credentials, authority or audit (approvals, the guard's audit, pairing, bootstrap)
# holds configuration the instance needs exactly as it is, and is never touched.
MEMORY_STORES = ("turn-idempotency.db", "colonymind-comms.db", "colonymind-facts.db", "colonymind-commitments.db",
                 "colonymind-self-model.db", "mind.db", "contacts.db", "colonymind-contacts.db", "colonymind-affect.db",
                 "colonymind-preferences.db", "colonymind-expectations.db", "initiatives.db",
                 "colonymind-action-journal.db", "colonymind-feedback.db", "colonymind-learning-feedback.db",
                 "insights.db", "colonymind-situation.db")
# Columns that name whose a row is: a value found in the row is sealed in that person's scope.
SCOPE_COLUMNS = ("contact_id", "person_id", "owner_id", "subject_id")
# Stores whose words are the owner's or the mind's own (a message's recipient is not its author): the owner's scope.
OWNER_STORES = ("initiatives.db", "mind.db")
SHADOW_SUFFIXES = ("_data", "_idx", "_docsize", "_config", "_content")
# Derived from turn_sources and rebuilt for every rewritten turn (the stemmed keyword index follows its source by
# trigger: colonymind.turns.keywords).
DERIVED_TABLES = {(LEDGER_NAME, "turn_source_search"), (LEDGER_NAME, "turn_sources"), (LEDGER_NAME, PENDING_TABLE),
                  (LEDGER_NAME, "turn_source_stems"), (LEDGER_NAME, "turn_source_stem_chunks"),
                  (LEDGER_NAME, "turn_source_stem_pending")}
BATCH = 200


def _shape(value: str) -> str:
    """A value's character classes and run lengths (never the value): ``L4_D6`` and the like."""
    def bucket(n: int) -> str:
        if n <= 12:
            return str(n)
        for low, high in ((13, 19), (20, 31), (32, 47), (48, 63), (64, 127)):
            if low <= n <= high:
                return f"{low}-{high}"
        return "128+"
    out = []
    runs = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", value)
    for run in runs[:8]:
        if not run.isalnum():
            out.append(run if run in "-_.:/+=@ " else "?")
            continue
        cls = "D" if run.isdigit() else "U" if run.isupper() else "L" if run.islower() else "M"
        out.append(f"{cls}{bucket(len(run))}")
    return "".join(out) + ("..." if len(runs) > 8 else "")


@dataclass
class Report:
    dry_run: bool = False
    turns_scanned: int = 0
    turns_changed: int = 0
    messages_changed: int = 0
    rules: Counter = field(default_factory=Counter)
    shapes: Counter = field(default_factory=Counter)
    lineage_pairs: int = 0
    lineage_rows: int = 0
    lineage_turns: int = 0
    stores: Counter = field(default_factory=Counter)
    unsealed: Counter = field(default_factory=Counter)
    withheld: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"dry_run": self.dry_run,
                "turn_sources": {"turns_scanned": self.turns_scanned, "turns_changed": self.turns_changed,
                                 "messages_changed": self.messages_changed},
                "rules": dict(self.rules), "shapes": dict(self.shapes.most_common(20)),
                "lineage": {"pairs": self.lineage_pairs, "rows_repointed": self.lineage_rows,
                            "turns_repointed": self.lineage_turns},
                "stores": [{"db": db, "table": table, "column": column, "values": count}
                           for (db, table, column), count in sorted(self.stores.items()) if count],
                "left_in_append_only_tables": [{"db": db, "table": table, "column": column, "rows": count}
                                               for (db, table, column), count in sorted(self.unsealed.items())],
                "withheld": self.withheld}


def _tables(conn: sqlite3.Connection) -> List[str]:
    # Owner goals and their immutable audit/receipts belong to OwnerGoalsStore.
    # Generic lineage and sealing walkers must not read or rewrite these tables.
    return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                           "AND name NOT LIKE 'sqlite_%'")
            if not row[0].startswith("owner_goal_")
            and not any(row[0].endswith(suffix) for suffix in SHADOW_SUFFIXES)]


def _columns(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str, int]]:
    try:
        return [(row[1], str(row[2] or "").upper(), int(row[5] or 0))
                for row in conn.execute(f'PRAGMA table_info("{table}")')]
    except sqlite3.DatabaseError:
        return []


def _text_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Columns that may hold text: declared text, or untyped (FTS and dynamically typed tables)."""
    return [name for name, kind, _ in _columns(conn, table)
            if not kind or "CHAR" in kind or "TEXT" in kind or "CLOB" in kind or kind == "JSON"]


def _databases(state_dir: Path) -> List[Path]:
    return [state_dir / name for name in MEMORY_STORES if (state_dir / name).is_file()]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA secure_delete=ON")
    return conn


def _transformed(original: str, transform: Callable[[str], str]) -> str:
    """``original`` with ``transform`` applied to its words. A JSON object or array is decoded first and the
    transform applied to its string leaves (so a value is its bytes, never the escapes that encode it), then
    written back in the original's separators; anything else is transformed as text. Unchanged words come back
    as the very same string."""
    if original.lstrip()[:1] in ("{", "["):
        try:
            parsed = json.loads(original)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            changed = map_leaves(parsed, transform)
            if changed == parsed:
                return original
            compact = ", " not in original and ": " not in original
            return json.dumps(changed, ensure_ascii="\\u" in original,
                              separators=(",", ":") if compact else (", ", ": "))
    return transform(original)


class StoreRewriter:
    """One pass over an instance's stores with a text transform: the turn ledger first (digests, index, vectors,
    lineage pairs), then lineage, then every other text column."""

    def __init__(self, state_dir: Path, *, report: Report, dry_run: bool = False) -> None:
        self.state_dir = Path(state_dir)
        self.report = report
        self.dry_run = dry_run
        self.ledger_path = self.state_dir / LEDGER_NAME

    # -- 1. the turn ledger -----------------------------------------------------------------------------------

    def rewrite_turns(self, transform: Callable[[List[Dict[str, Any]], sqlite3.Row], List[Dict[str, Any]]],
                      *, where: str = "", args: Tuple = (), lineage: bool = False) -> None:
        """``transform`` over every turn (``where`` narrows them); a changed turn is stored with its digests and
        its identity pairs. ``lineage`` counts the pass as re-pointing, not as sealing."""
        if not self.ledger_path.exists():
            return
        ledger = None
        if not self.dry_run:
            from colonymind.turns.idempotency import TurnIdempotencyLedger
            ledger = TurnIdempotencyLedger(self.ledger_path)   # makes sure the ledger's tables exist
            with closing(_connect(self.ledger_path)) as conn, conn:
                conn.execute(f"CREATE TABLE IF NOT EXISTS {PENDING_TABLE} (old TEXT PRIMARY KEY, new TEXT NOT NULL)")
        cursor = 0
        while True:
            with closing(_connect(self.ledger_path)) as conn:
                rows = conn.execute(f"SELECT rowid AS rid, * FROM turn_sources WHERE rowid > ? {where} "
                                    f"ORDER BY rowid LIMIT {BATCH}", (cursor, *args)).fetchall()
            if not rows:
                return
            for row in rows:
                cursor = row["rid"]
                self.report.turns_scanned += 0 if lineage else 1
                try:
                    messages = json.loads(row["messages_json"])
                except ValueError:
                    continue
                if not isinstance(messages, list):
                    continue
                changed = transform(messages, row)
                if changed == messages:
                    continue
                if lineage:
                    self.report.lineage_turns += 1
                else:
                    self.report.turns_changed += 1
                    self.report.messages_changed += sum(1 for a, b in zip(messages, changed) if a != b)
                if not self.dry_run:
                    self._store_turn(ledger, row, messages, changed)

    def _store_turn(self, ledger: Any, row: sqlite3.Row, before: List[Dict[str, Any]],
                    after: List[Dict[str, Any]]) -> None:
        from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash
        from colonymind.turns.source_vectors import enqueue as enqueue_vectors
        session, turn_id = row["session_id"], row["turn_id"]
        pairs = {old: new for old, new in ((source_message_hash(session, a), source_message_hash(session, b))
                                            for a, b in zip(before, after)) if old != new}
        pairs[canonical_turn_digest(before)] = canonical_turn_digest(after)
        envelope = {"contact_id": row["contact_id"], "session_id": session, "scope": row["scope"],
                    "occurred_at": row["occurred_at"]}
        content = canonical_turn_digest({"messages": after, **envelope})
        pairs[row["content_sha256"]] = content
        pairs[row["turn_id"]] = turn_id
        encoded = json.dumps(after, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with closing(_connect(self.ledger_path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute("UPDATE turn_sources SET turn_id=?, messages_json=?, content_sha256=? "
                                   "WHERE turn_id=? AND messages_json=?",
                                   (turn_id, encoded, content, row["turn_id"], row["messages_json"]))
            if not updated.rowcount:
                return  # changed underneath us: the next run sees the new row
            conn.execute("DELETE FROM turn_source_search WHERE turn_id=?", (row["turn_id"],))
            if turn_id != row["turn_id"]:
                conn.execute("DELETE FROM source_vector_jobs WHERE turn_id=?", (row["turn_id"],))
            ledger._index_messages(conn, turn_id, after)
            enqueue_vectors(conn, turn_id)
            # The ingestion row's digest was over the envelope as it arrived: it must not stay (a digest of a
            # short value is a way back to it). A replay of that very envelope now answers as a conflict.
            conn.execute("UPDATE turn_ingestion SET content_sha256=? WHERE turn_id=?",
                         (canonical_turn_digest({"rewritten_source": row["turn_id"], "content": content}),
                          row["turn_id"]))
            for old, new in pairs.items():
                if old != new:
                    conn.execute(f"INSERT INTO {PENDING_TABLE}(old, new) VALUES (?, ?) ON CONFLICT(old) DO UPDATE "
                                 "SET new=excluded.new", (old, new))

    # -- 2. lineage -----------------------------------------------------------------------------------------

    def repoint_lineage(self) -> None:
        """Replace every old identity with its new one in every store, to a fixpoint (a child turn whose source
        references changed has a new version and content digest of its own)."""
        if self.dry_run or not self.ledger_path.exists():
            return
        for _ in range(16):
            with closing(_connect(self.ledger_path)) as conn:
                try:
                    pairs = [(row["old"], row["new"]) for row in conn.execute(f"SELECT old, new FROM {PENDING_TABLE}")]
                except sqlite3.OperationalError:
                    return
            if not pairs:
                return
            self.report.lineage_pairs += len(pairs)
            children = self._repoint_child_turns(pairs)
            for path in _databases(self.state_dir):
                self._repoint_database(path, pairs)
            with closing(_connect(self.ledger_path)) as conn, conn:
                conn.executemany(f"DELETE FROM {PENDING_TABLE} WHERE old=? AND new=?", pairs)
            if not children:
                return

    def _repoint_child_turns(self, pairs: List[Tuple[str, str]]) -> int:
        """Turns whose messages cite an old identity (a source reference to a rewritten parent): rewritten through
        ``rewrite_turns`` so their own digests and pairs follow."""
        mapping = dict(pairs)

        def repoint(messages: List[Dict[str, Any]], _row: sqlite3.Row) -> List[Dict[str, Any]]:
            text = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for old, new in mapping.items():
                text = text.replace(old, new)
            return json.loads(text)
        before = self.report.lineage_turns
        for start in range(0, len(pairs), 50):
            chunk = pairs[start:start + 50]
            clause = " AND (" + " OR ".join("instr(messages_json, ?) > 0" for _ in chunk) + ")"
            self.rewrite_turns(repoint, where=clause, args=tuple(old for old, _ in chunk), lineage=True)
        return self.report.lineage_turns - before

    def _repoint_database(self, path: Path, pairs: List[Tuple[str, str]]) -> None:
        with closing(_connect(path)) as conn, conn:
            for table in _tables(conn):
                if (path.name, table) in DERIVED_TABLES or self._is_virtual(conn, table):
                    continue
                for column in _text_columns(conn, table):
                    for old, new in pairs:
                        try:
                            cursor = conn.execute(f'UPDATE "{table}" SET "{column}"=replace("{column}", ?, ?) '
                                                  f'WHERE instr("{column}", ?) > 0', (old, new, old))
                        except sqlite3.DatabaseError:
                            break
                        self.report.lineage_rows += max(0, cursor.rowcount)

    @staticmethod
    def _is_virtual(conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()
        return bool(row and row[0] and row[0].upper().startswith("CREATE VIRTUAL"))

    def index_derived(self) -> None:
        """Index what the ledger's derived indexes queued while identities were re-pointed and values rewritten (a
        claim moved with its message is queued for the claim key index): once every rewrite is done, so what the
        pass left untouched is found as before, not only after the ledger next opens."""
        if self.dry_run or not self.ledger_path.exists():
            return
        from colonymind.turns.idempotency import TurnIdempotencyLedger
        TurnIdempotencyLedger(self.ledger_path).index_pending()

    # -- 3. every other text column ------------------------------------------------------------------------------

    def walk(self, transform: Callable[[str, Dict[str, Any]], str], *, prefilter: Optional[str] = None) -> None:
        """``transform(text, row)`` over every text column of every store (turn_sources and its index excepted);
        ``prefilter`` skips values that do not contain it."""
        for path in _databases(self.state_dir):
            try:
                with closing(_connect(path)) as conn:
                    tables = _tables(conn)
            except sqlite3.DatabaseError as error:
                logger.warning("secrets migration skipped %s (%s)", path.name, type(error).__name__)
                continue
            for table in tables:
                if (path.name, table) in DERIVED_TABLES:
                    continue
                try:
                    self._walk_table(path, table, transform, prefilter)
                except VaultUnavailable:
                    raise   # strict sealing: stop here, change nothing more, resume next time
                except sqlite3.DatabaseError as error:
                    logger.warning("secrets migration skipped %s:%s (%s)", path.name, table, type(error).__name__)

    def _walk_table(self, path: Path, table: str, transform: Callable[[str, Dict[str, Any]], str],
                    prefilter: Optional[str]) -> None:
        with closing(_connect(path)) as conn:
            columns = _columns(conn, table)
            texts = _text_columns(conn, table)
            keys = [name for name, _, pk in sorted(columns, key=lambda item: item[2]) if pk]
            try:
                conn.execute(f'SELECT rowid FROM "{table}" LIMIT 1').fetchall()
                keys = ["rowid"]
            except sqlite3.DatabaseError:
                if not keys:
                    return
        names = [name for name, _, _ in columns]
        for column in texts:
            self._walk_column(path, table, column, keys, names, transform, prefilter)

    def _walk_column(self, path: Path, table: str, column: str, keys: List[str], names: List[str],
                     transform: Callable[[str, Dict[str, Any]], str], prefilter: Optional[str]) -> None:
        key_sql = ", ".join(f'"{key}"' if key != "rowid" else "rowid" for key in keys)
        extra = [name for name in SCOPE_COLUMNS if name in names]
        select = ", ".join([f"{key_sql}", f'"{column}"', *(f'"{name}"' for name in extra)])
        condition = f'"{column}" IS NOT NULL AND length("{column}") >= 8'
        args: Tuple = ()
        if prefilter:
            condition += f' AND instr("{column}", ?) > 0'
            args = (prefilter,)
        last: Optional[Tuple] = None
        while True:
            with closing(_connect(path)) as conn:
                if keys == ["rowid"]:
                    rows = conn.execute(f'SELECT {select} FROM "{table}" WHERE {condition} AND rowid > ? '
                                        f'ORDER BY rowid LIMIT {BATCH}', (*args, last[0] if last else -1)).fetchall()
                else:
                    rows = conn.execute(f'SELECT {select} FROM "{table}" WHERE {condition}', args).fetchall()
            if not rows:
                return
            updates = []
            for row in rows:
                value = row[len(keys)]
                last = tuple(row[index] for index in range(len(keys)))
                if not isinstance(value, str):
                    continue
                context = {name: row[len(keys) + 1 + index] for index, name in enumerate(extra)}
                context["_store"] = path.name
                changed = _transformed(value, lambda text, context=context: transform(text, context))
                if changed != value:
                    self.report.stores[(path.name, table, column)] += 1
                    updates.append((changed, *last, value))
            if updates and not self.dry_run:
                where = " AND ".join(f'{"rowid" if key == "rowid" else chr(34) + key + chr(34)}=?' for key in keys)
                try:
                    with closing(_connect(path)) as conn, conn:
                        conn.executemany(f'UPDATE "{table}" SET "{column}"=? WHERE {where} AND "{column}"=?',
                                         updates)
                except sqlite3.DatabaseError:
                    # An append-only table (its triggers refuse every update) keeps what it holds: reported,
                    # never forced.
                    self.report.stores[(path.name, table, column)] -= len(updates)
                    self.report.unsealed[(path.name, table, column)] += len(updates)
            if keys != ["rowid"]:
                return


# -- the migration -------------------------------------------------------------------------------------------------


def _count_hits(text: str, report: Report, settings: SecretsSettings) -> bool:
    hits = find_secrets(text, rules=settings.rules)
    for hit in hits:
        report.rules[hit.rule] += 1
        report.shapes[_shape(hit.value)] += 1
    return bool(hits)


def _message_texts(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    yield block["text"]
                for segment in block.get("segments") or []:
                    if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                        yield segment["text"]


def legacy_backend_note(home: Optional[Path] = None) -> Dict[str, Any]:
    """What the removed connector-credential store was configured with: the backend's name and, for the file and
    keyring backends, the key names it held (never a value). Those values stay where they are."""
    home = Path(home or os.environ.get("COLONYMIND_HOME") or Path.home() / ".colonymind")
    note: Dict[str, Any] = {"backend": None, "key_names": [], "where": None}
    try:
        import yaml
        legacy = home / "config.yaml"
        if legacy.is_file():
            data = yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
            section = data.get("secrets") if isinstance(data, dict) else None
            note["backend"] = section.get("backend") if isinstance(section, dict) else None
    except Exception:
        note["backend"] = "unreadable"
    env_file = home / ".env"
    if env_file.is_file():
        try:
            names = [line.split("=", 1)[0].strip() for line in env_file.read_text(encoding="utf-8").splitlines()
                     if "=" in line and not line.lstrip().startswith("#")]
            note["key_names"] = sorted(name for name in names if name.startswith("connector/"))
            note["where"] = "the instance .env file" if note["key_names"] else None
        except OSError:
            pass
    if note["backend"] == "keyring":
        note["where"] = "the OS keyring (service colonymind-ai); probe with the keyring tool, values stay there"
    elif note["backend"] == "1password":
        note["where"] = "the 1Password vault named ColonyMind; its items stay there"
    return note


def hermes_transcripts(hermes_home: Optional[Path], settings: SecretsSettings, *,
                       limit: int = 400_000) -> Dict[str, Any]:
    """A read-only count of Hermes transcript messages that still hold a value (the database is opened
    immutable: no lock, no journal, nothing written)."""
    if hermes_home is None:
        return {"checked": False, "reason": "no Hermes home configured"}
    path = Path(hermes_home).expanduser() / "state.db"
    if not path.is_file():
        return {"checked": False, "reason": "no Hermes state database"}
    counted, sessions, scanned = 0, set(), 0
    try:
        conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
        try:
            for session, content in conn.execute("SELECT session_id, content FROM messages ORDER BY id DESC LIMIT ?",
                                                 (limit,)):
                scanned += 1
                if isinstance(content, str) and find_secrets(content, rules=settings.rules):
                    counted += 1
                    sessions.add(session)
        finally:
            conn.close()
    except sqlite3.DatabaseError as error:
        return {"checked": False, "reason": f"unreadable ({type(error).__name__})"}
    return {"checked": True, "messages_scanned": scanned, "messages_with_values": counted,
            "sessions_with_values": len(sessions),
            "next_step": ("Hermes offers no in-place scrub: `hermes sessions delete <id>` removes a session, "
                          "the owner's call") if counted else None}


def _owner_id() -> Optional[str]:
    try:
        from colonymind.identity import get_owner_contact_id
        return get_owner_contact_id() or None
    except Exception:
        return None


def migrate(state_dir: Optional[Path] = None, *, dry_run: bool = False, vault: Optional[Vault] = None,
            settings: Optional[SecretsSettings] = None, hermes_home: Optional[Path] = None,
            write_report: bool = True, owner_id: Optional[str] = None) -> Dict[str, Any]:
    """Seal every stored secret (or, with ``dry_run``, count them). Idempotent and resumable. Raises
    ``VaultUnavailable`` when the vault cannot take a value: nothing of that turn or row changed, what was sealed
    before stays sealed, and no completion marker is written (the next run resumes)."""
    from colonymind import get_state_dir
    state_dir = Path(state_dir or get_state_dir())
    owner = owner_id or _owner_id()
    settings = settings or current_settings()
    report = Report(dry_run=dry_run)
    rewriter = StoreRewriter(state_dir, report=report, dry_run=dry_run)
    vault = None if dry_run else (vault or get_vault(state_dir, key_store=settings.key_store))
    if vault is not None and not vault.available:
        raise VaultUnavailable("the vault key is unavailable; nothing was changed")
    started = ledger_stamp()

    def seal_turn(messages: List[Dict[str, Any]], row: sqlite3.Row) -> List[Dict[str, Any]]:
        if dry_run:
            counted = []
            for message in messages:
                hit = False
                if isinstance(message, dict):
                    for text in _message_texts(message.get("content")):
                        hit = _count_hits(text, report, settings) or hit
                counted.append(dict(message, _counted=True) if hit else message)
            return counted
        return redact_sensitive_messages(messages, scope=Scope(row["contact_id"], row["session_id"], row["turn_id"]),
                                         vault=vault, settings=settings, strict=True)
    rewriter.rewrite_turns(seal_turn)
    rewriter.repoint_lineage()

    def seal_value(text: str, row: Mapping[str, Any]) -> str:
        if dry_run:
            if _count_hits(text, report, settings):
                return text + "\x00"  # counted as changed, never written in a dry run
            return text
        if row.get("_store") in OWNER_STORES:
            scope = owner or "unscoped"
        else:
            scope = next((str(row[name]) for name in SCOPE_COLUMNS if row.get(name)), "unscoped")
        return seal_text(text, scope=Scope(contact_id=scope), vault=vault, settings=settings, strict=True)
    rewriter.walk(seal_value)
    rewriter.index_derived()
    from colonymind.owner_goals.store import OwnerGoalsStore
    report.stores.update(OwnerGoalsStore.seal_existing(state_dir / "initiatives.db", vault=vault, dry_run=dry_run))
    backups = state_dir / "backups"
    result = report.as_dict()
    result.update({
        "version": 1, "started_at": started, "completed_at": ledger_stamp(),
        "backups_predating": len(list(backups.iterdir())) if backups.is_dir() else 0,
        "backups_note": "nightly backups taken before this run still hold the values until they roll off",
        "legacy_backend": legacy_backend_note(),
        "hermes_transcripts": hermes_transcripts(hermes_home, settings),
    })
    if not dry_run:
        vault.set_marker(MARKER, result["completed_at"])
        if write_report:
            path = state_dir / REPORT_NAME
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(path, 0o600)
    return result


def migrated(state_dir: Optional[Path] = None, *, vault: Optional[Vault] = None) -> bool:
    from colonymind import get_state_dir
    vault = vault or get_vault(Path(state_dir or get_state_dir()))
    return vault.marker(MARKER) is not None


def migrate_on_start(state_dir: Path, *, settings: Optional[SecretsSettings] = None,
                     hermes_home: Optional[Path] = None) -> Optional[threading.Thread]:
    """At the first start: the migration in a background thread, once (the marker guards it)."""
    settings = settings or current_settings()
    if not settings.migrate_on_start or not settings.capture_enabled:
        return None
    vault = get_vault(state_dir, key_store=settings.key_store)
    if vault.marker(MARKER) is not None:
        return None

    def run() -> None:
        try:
            result = migrate(state_dir, vault=vault, settings=settings, hermes_home=hermes_home)
            logger.info("secrets migration done: %d turns sealed, %d values in other stores",
                        result["turn_sources"]["turns_changed"], sum(item["values"] for item in result["stores"]))
        except Exception as error:
            logger.warning("secrets migration stopped (%s); it resumes at the next start", type(error).__name__)
    thread = threading.Thread(target=run, name="colonymind-secrets-migrate", daemon=True)
    thread.start()
    return thread


# -- release ------------------------------------------------------------------------------------------------------


def release(ref: str, state_dir: Optional[Path] = None, *, vault: Optional[Vault] = None) -> Dict[str, Any]:
    """A false positive: the value goes back into memory wherever its reference stood, and is never sealed again
    in that scope. Counts only in the answer."""
    from colonymind import get_state_dir
    state_dir = Path(state_dir or get_state_dir())
    vault = vault or get_vault(state_dir)
    value = vault.release(ref)
    report = Report()
    rewriter = StoreRewriter(state_dir, report=report)
    values = {ref: value}

    def restore(messages: List[Dict[str, Any]], _row: sqlite3.Row) -> List[Dict[str, Any]]:
        # Every field sealing reached (``seal.map_message``): the content, metadata, an observation's arguments.
        return [map_message(message, lambda text: unseal_text(text, values)) if isinstance(message, dict)
                else message for message in messages]
    rewriter.rewrite_turns(restore, where=" AND instr(messages_json, ?) > 0", args=(ref,))
    rewriter.repoint_lineage()
    rewriter.walk(lambda text, _row: unseal_text(text, values), prefilter=ref)
    rewriter.index_derived()
    return {"ref": ref, "turns": report.turns_changed, "messages": report.messages_changed,
            "other_rows": sum(report.stores.values())}


__all__ = ["MARKER", "REPORT_NAME", "Report", "StoreRewriter", "hermes_transcripts", "legacy_backend_note", "migrate",
           "migrate_on_start", "migrated", "release"]
