"""Generic schema migration runner for ColonyMind SQLite stores.

Discovers numbered .sql files in a migrations directory, tracks which
have been applied via a ``schema_version`` table in the target database,
and applies unapplied ones in order.  Works with both sync (sqlite3)
and async (aiosqlite) connections.

Usage (sync)::

    import sqlite3
    conn = sqlite3.connect("colonymind-contacts.db")
    applied = run_migrations_sync(conn, Path("contacts/migrations"))

Usage (async)::

    import aiosqlite
    db = await aiosqlite.connect("colonymind-contacts.db")
    applied = await run_migrations(db, Path("contacts/migrations"))

Migration filenames must match ``NNN_description.sql`` (e.g.
``001_initial_schema.sql``).  The numeric prefix determines order.
Each file runs as one transaction together with its ``schema_version``
row (``_script``): either both land or neither does, so a failure, a
kill or a locked database part way leaves the store at the previous
version for a rerun, never with a migration applied and unrecorded. A
file therefore holds no ``BEGIN``/``COMMIT`` of its own. A file that
drops a column needs SQLite >= 3.35; both runners check that before
applying anything.

A release's one-time change to rows that no SQL file can express (it
reads each row's history and hands back what the row fed) is a data
step below; the owner of the rows runs it when it opens its store, and
running it again changes nothing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

_VERSION_DDL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version     TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_MIGRATION_RE = re.compile(r"^(\d+)_.+\.sql$")


DROP_COLUMN_SQLITE = (3, 35)
_DROP_COLUMN_RE = re.compile(r"\bDROP\s+COLUMN\b", re.IGNORECASE)


def _check_sqlite(pending: list[tuple[str, Path]]) -> None:
    """Refuse, before any file runs, a migration SQLite cannot apply (ALTER TABLE DROP COLUMN)."""
    if sqlite3.sqlite_version_info >= DROP_COLUMN_SQLITE:
        return
    for version, path in pending:
        if _DROP_COLUMN_RE.search(path.read_text()):
            raise RuntimeError(
                f"migration {version} ({path.name}) needs SQLite >= 3.35 (ALTER TABLE DROP COLUMN); "
                f"this Python links SQLite {sqlite3.sqlite_version}")


def _script(sql: str, *, table: str, version: str, filename: str) -> str:
    """One migration file and its version row as a single transaction for ``executescript``."""
    def quoted(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"
    return (f"BEGIN;\n{sql}\n;\nINSERT INTO {table}(version, filename) "
            f"VALUES ({quoted(version)}, {quoted(filename)});\nCOMMIT;\n")


def _discover(migrations_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(version, path)`` pairs sorted by numeric prefix."""
    found: list[tuple[int, str, Path]] = []
    if not migrations_dir.is_dir():
        return []
    for p in migrations_dir.iterdir():
        m = _MIGRATION_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), m.group(1).zfill(3), p))
    found.sort(key=lambda t: t[0])
    return [(v, p) for _, v, p in found]


# ── Sync API (sqlite3) ──────────────────────────────────────────────────


def run_migrations_sync(
    conn: sqlite3.Connection,
    migrations_dir: Path,
    *,
    table: str = "schema_version",
) -> list[str]:
    """Apply unapplied migrations using a sync sqlite3 connection.

    Returns the list of newly applied version strings.
    """
    ddl = _VERSION_DDL.replace("schema_version", table)
    conn.executescript(ddl)
    conn.commit()

    cur = conn.execute(f"SELECT version FROM {table}")  # noqa: S608
    applied = {row[0] for row in cur.fetchall()}

    pending = [(version, path) for version, path in _discover(migrations_dir) if version not in applied]
    _check_sqlite(pending)
    newly_applied: list[str] = []

    for version, path in pending:
        sql = path.read_text()
        logger.info("Applying migration %s (%s)", version, path.name)
        try:
            conn.executescript(_script(sql, table=table, version=version, filename=path.name))
        except BaseException:
            logger.error("Migration %s failed (%s)", version, path.name)
            conn.rollback()
            raise
        newly_applied.append(version)

    if newly_applied:
        logger.info(
            "Applied %d migration(s): %s",
            len(newly_applied),
            ", ".join(newly_applied),
        )
    return newly_applied


# ── Async API (aiosqlite) ───────────────────────────────────────────────


async def run_migrations(
    db: "aiosqlite.Connection",
    migrations_dir: Path,
    *,
    table: str = "schema_version",
) -> list[str]:
    """Apply unapplied migrations using an async aiosqlite connection.

    Returns the list of newly applied version strings.
    """
    ddl = _VERSION_DDL.replace("schema_version", table)
    await db.executescript(ddl)
    await db.commit()

    async with db.execute(f"SELECT version FROM {table}") as cur:  # noqa: S608
        rows = await cur.fetchall()
    applied = {row[0] for row in rows}

    pending = [(version, path) for version, path in _discover(migrations_dir) if version not in applied]
    _check_sqlite(pending)
    newly_applied: list[str] = []

    for version, path in pending:
        sql = path.read_text()
        logger.info("Applying migration %s (%s)", version, path.name)
        try:
            await db.executescript(_script(sql, table=table, version=version, filename=path.name))
        except BaseException:
            logger.error("Migration %s failed (%s)", version, path.name)
            await db.rollback()
            raise
        newly_applied.append(version)

    if newly_applied:
        logger.info(
            "Applied %d migration(s): %s",
            len(newly_applied),
            ", ".join(newly_applied),
        )
    return newly_applied


def applied_versions_sync(
    conn: sqlite3.Connection,
    *,
    table: str = "schema_version",
) -> set[str]:
    """Return the set of already-applied version strings (sync)."""
    try:
        cur = conn.execute(f"SELECT version FROM {table}")  # noqa: S608
        return {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()


# ── Data steps: asks from before answers were read from the owner's own turn ──

UPGRADE_BY = "upgrade"                  # ``by`` of what a data step records
VERDICT_CLEARED = "verdict_cleared"     # the history row of a verdict a data step cleared
# The asks an answer could leave a verdict on (an ask to act, to message, to pursue a goal): every note ask (a
# standing grant, an identity link, a check-in or people change, a wish) settles with no verdict.
VERDICT_KINDS = ["task", "message", "goal"]
ROWS_READ = 100000                      # rows one step reads (my choice: every ask a store keeps)
_FORMED = frozenset({"decided_ask", "asked"})
# The history rows that replace an ask's verdict after its answer settled it: a rating (the owner's, or the mind's
# own score) sets a new one, and this step's clearing removes it. Either makes the answer's verdict no longer the
# one the row holds.
_REVERDICTS = frozenset({"rated", VERDICT_CLEARED})
# The notice that listed open asks by their code, before each ask was its own message.
NOTICE_TYPE = "ask_notice"
# How far before the oldest open ask a listed code is looked up: as the answer route reads such a notice.
_CODE_DAYS = 4


def _aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _details(item: Any) -> dict:
    return item.details if isinstance(getattr(item, "details", None), dict) else {}


def _ordered(history: list) -> list:
    return sorted(history, key=lambda item: (_aware(item.timestamp) or datetime.min.replace(tzinfo=timezone.utc),
                                             item.id or 0))


def settling_row(history: list) -> Any:
    """The history row that settled an ask: the first approval naming who answered (``queued``) or outcome after the
    ask formed; None while it waits."""
    ordered = _ordered(history)
    start = next((index for index, item in enumerate(ordered) if item.action in _FORMED), 0)
    for item in ordered[start:]:
        if (item.action == "queued" and _details(item).get("by")) or item.action.startswith("outcome_"):
            return item
    return None


def verdict_row(history: list) -> Any:
    """The history row an ask's current verdict came from: the last rating or clearing after its settlement
    (``_REVERDICTS``), else the settlement itself; None while it waits."""
    ordered = _ordered(history)
    settled = settling_row(ordered)
    if settled is None:
        return None
    later = [item for item in ordered[ordered.index(settled) + 1:] if item.action in _REVERDICTS]
    return later[-1] if later else settled


def clear_operator_verdicts(store: Any, *, withdraw: Optional[Callable[[Any], None]] = None,
                            now: Optional[datetime] = None) -> list[str]:
    """An ask whose answer came from anyone but the owner in their own turn (the CLI, a credential, a script at the
    console: ``outcomes.owner_answer`` of its settling row) keeps its outcome and loses the verdict an earlier release
    took from that answer. Only that verdict: when a later history row replaced it (``verdict_row``: the owner's
    rating, the mind's own score, or this step's own clearing), the verdict the row holds is not the answer's and is
    kept, so a rating given before or after the upgrade survives every later start, and an ask this step cleared is
    never cleared again. ``withdraw(row)`` takes the verdict's feedback back first and raises when it cannot (the
    feedback store locked, or not open at this start): that ask then keeps its verdict, which is what marks it for
    the next run, and the step goes on with the others, so no verdict is cleared while its feedback still counts.
    Each ask cleared gets a ``verdict_cleared`` history row naming the verdict and who answered; the ids cleared and
    those left for the next run are logged. Returns the ids cleared."""
    from colonymind.mind.outcomes import owner_answer
    from colonymind.util.temporal import now_utc
    now = now or now_utc()
    cleared: list[str] = []
    waiting: list[str] = []
    for row in store.intentions(kind=VERDICT_KINDS, limit=ROWS_READ):
        if not row.ask_code or not row.verdict:
            continue
        history = store.get_history(row.id, limit=500)
        settled = settling_row(history)
        if settled is None or verdict_row(history) is not settled:
            continue
        details = _details(settled)
        if owner_answer(details.get("by"), details) is not False:
            continue
        if withdraw is not None:
            try:
                withdraw(row)
            except Exception as error:
                logger.warning("upgrade: the feedback of ask %s was not withdrawn (%s); its verdict stays until a "
                               "start can withdraw it", row.id, type(error).__name__)
                waiting.append(row.id)
                continue
        store.transition(row.id, row.status, action=VERDICT_CLEARED, at=now, verdict=None,
                         details={"by": UPGRADE_BY, "verdict": row.verdict, "answered_by": details.get("by"),
                                  "why": "an answer the owner did not give in their own turn is no verdict"})
        cleared.append(row.id)
    if cleared:
        logger.info("upgrade: cleared the verdict an operator's answer left on %d ask(s): %s", len(cleared),
                    ", ".join(cleared))
    if waiting:
        logger.warning("upgrade: %d ask(s) keep an operator's verdict until their feedback can be withdrawn: %s",
                       len(waiting), ", ".join(waiting))
    return cleared


def _listed_codes(text: str, label: str) -> list[str]:
    """The codes a notice lists under ``label``, in its order (``- CODE: sentence`` lines)."""
    codes: list[str] = []
    inside = False
    for line in str(text or "").splitlines():
        if line == f"{label}:":
            inside = True
        elif inside and line.startswith("- "):
            codes.append(line[2:].split(":", 1)[0].strip().upper())
        elif inside:
            inside = False
    return codes


def bind_earlier_notices(store: Any, *, now: Optional[datetime] = None) -> list[str]:
    """Each open ask's last notice of the earlier format (a sent ``ask_notice`` listing asks by code, which this
    release no longer reads that way) is recorded as a list binding on the ask: an ``ask_shown`` history row with the
    notice's platform, chat, delivery id and time and the asks it listed in its order (a code no row takes keeps its
    place, so numbers keep their meaning). A quoted reply or a reaction to that notice then binds as the notice showed
    it, whatever the notice text becomes. Recorded once per ask; a notice whose delivery id or chat is unknown names
    nothing and is skipped. Returns the ids bound."""
    from colonymind.mind.asks_report import ASK_SHOWN, NEEDS_OK_1_11
    from colonymind.mind.tick import split_target
    open_rows = [row for row in store.intentions(status=["asked"], limit=500) if row.ask_code]
    created = [at for at in (_aware(row.created_at) for row in open_rows) if at is not None]
    if not open_rows or not created:
        return []
    since = min(created)
    coded: dict[str, list] = {}
    for row in store.intentions(since=since - timedelta(days=_CODE_DAYS), limit=ROWS_READ):
        if row.ask_code:
            coded.setdefault(str(row.ask_code).upper(), []).append(row)
    listings = []
    for notice in store.intentions(status=["sent"], kind=["message"], types=[NOTICE_TYPE], since=since,
                                   limit=500):
        made, delivered = _aware(notice.created_at), _aware(notice.completed_at)
        context = notice.context if isinstance(notice.context, dict) else {}
        if not notice.hermes_ref or delivered is None:
            continue
        ids = []
        for code in _listed_codes(str(context.get("text") or ""), NEEDS_OK_1_11):
            match = next((row for row in coded.get(code, [])
                          if made is None or _aware(row.created_at) is None or _aware(row.created_at) <= made), None)
            ids.append(match.id if match is not None else "")
        target = next((_details(item).get("target") for item in store.get_history(notice.id, limit=50)
                       if item.action == "sending" and _details(item).get("target")), "")
        platform, chat = split_target(str(target or ""))
        if ids and platform and chat:
            listings.append((delivered, notice, platform, chat, ids))
    listings.sort(key=lambda item: item[0], reverse=True)
    bound: list[str] = []
    for row in open_rows:
        history = store.get_history(row.id, limit=500)
        if any(item.action == ASK_SHOWN and _details(item).get("by") == UPGRADE_BY for item in history):
            continue
        listing = next((item for item in listings if row.id in item[4]), None)
        if listing is None:
            continue
        delivered, notice, platform, chat, ids = listing
        store.transition(row.id, row.status, action=ASK_SHOWN, at=delivered,
                         details={"platform": platform, "chat": chat, "message_id": str(notice.hermes_ref),
                                  "at": delivered.isoformat(), "list": ids, "by": UPGRADE_BY, "notice": notice.id})
        bound.append(row.id)
    if bound:
        logger.info("upgrade: recorded the earlier notice of %d open ask(s) as a list binding: %s", len(bound),
                    ", ".join(bound))
    return bound
