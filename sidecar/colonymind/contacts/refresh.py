"""Durable per-contact digest invalidation and bounded background refresh."""

from contextlib import closing
from datetime import datetime, timezone

DEBOUNCE_SECONDS = 30
REFRESHES_PER_HOUR = 60
RETRY_SECONDS = 3600
MAX_RETRY_SECONDS = 24 * 3600


def initialize(conn):
    # These jobs commit with the fact change, including owner-ratified changes,
    # re-extraction and erasure. A crash between databases cannot lose a job.
    conn.execute("""CREATE TABLE IF NOT EXISTS contact_digest_jobs (
        contact_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, changed_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS contact_digest_refreshes (
        id INTEGER PRIMARY KEY, started_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS contact_digest_retries (
        contact_id TEXT PRIMARY KEY, last_attempt REAL NOT NULL,
        failures INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0)""")
    for action, ref in (("INSERT", "NEW"), ("UPDATE", "OLD"), ("DELETE", "OLD")):
        event = action
        condition = ""
        if action == "UPDATE":
            # Restatements move the anchor and its evidence without changing
            # the fact. Only lifecycle changes invalidate an existing claim.
            fields = ("superseded_by", "retracted_by", "valid_from", "valid_to")
            event += " OF " + ", ".join(fields)
            condition = "WHEN " + " OR ".join(f"OLD.{field} IS NOT NEW.{field}" for field in fields)
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS contact_digest_{action.lower()}
            AFTER {event} ON source_claims {condition} BEGIN
            INSERT INTO contact_digest_jobs(contact_id, revision, changed_at)
            SELECT contact_id, 1, (julianday('now') - 2440587.5) * 86400
            FROM turn_sources WHERE turn_id = {ref}.turn_id AND scope = 'person'
            ON CONFLICT(contact_id) DO UPDATE SET revision = revision + 1,
                changed_at = excluded.changed_at;
            END""")


async def invalidate_pending(ledger, store, *, owner_id):
    """Transfer invalidations to the contact store, acknowledging only the revision read."""
    marker = getattr(store, "mark_digest_stale", None)
    if ledger is None or not callable(marker):
        return
    with closing(ledger._connect()) as conn:
        jobs = conn.execute("SELECT contact_id, revision, changed_at FROM contact_digest_jobs "
                            "ORDER BY changed_at LIMIT 200").fetchall()
    for contact_id, revision, changed_at in jobs:
        if contact_id != owner_id:
            stamp = datetime.fromtimestamp(changed_at, timezone.utc).isoformat()
            await marker(contact_id, changed_at=stamp)
        with closing(ledger._connect()) as conn, conn:
            conn.execute("DELETE FROM contact_digest_jobs WHERE contact_id = ? AND revision = ?",
                         (contact_id, revision))


def reserve_refresh(ledger, now, *, contact_id=None):
    """Charge attempts, including failed writes, against a persistent rolling hour."""
    if ledger is None:
        return False
    with closing(ledger._connect()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM contact_digest_refreshes WHERE started_at <= ?", (now - 3600,))
        count = conn.execute("SELECT count(*) FROM contact_digest_refreshes").fetchone()[0]
        if count >= REFRESHES_PER_HOUR:
            return False
        conn.execute("INSERT INTO contact_digest_refreshes(started_at) VALUES (?)", (now,))
        if contact_id is not None:
            _record_attempt(conn, contact_id, now)
    return True


def _record_attempt(conn, contact_id, now):
    conn.execute("INSERT INTO contact_digest_retries(contact_id, last_attempt) VALUES (?, ?) "
                 "ON CONFLICT(contact_id) DO UPDATE SET last_attempt = excluded.last_attempt",
                 (contact_id, now))


def record_attempt(ledger, contact_id, now):
    """Nightly attempts share the background worker's durable queue ordering."""
    if ledger is not None:
        with closing(ledger._connect()) as conn, conn:
            _record_attempt(conn, contact_id, now)


def finish_refresh(ledger, contact_id, now, *, failed=False):
    """Keep failures queued with exponential backoff; a successful write resets it."""
    if ledger is None:
        return
    with closing(ledger._connect()) as conn, conn:
        row = conn.execute("SELECT failures FROM contact_digest_retries WHERE contact_id = ?",
                           (contact_id,)).fetchone()
        failures = (row[0] if row else 0) + 1 if failed else 0
        delay = min(MAX_RETRY_SECONDS, RETRY_SECONDS * 2 ** min(failures - 1, 5)) if failed else 0
        conn.execute("UPDATE contact_digest_retries SET failures = ?, retry_at = ? WHERE contact_id = ?",
                     (failures, now + delay if failed else 0, contact_id))


async def refresh_candidates(ledger, store, *, owner_id, now, limit, quiet_seconds=0):
    """Read every stale page before writes, prioritizing unattempted then least-recently attempted work."""
    reader = getattr(store, "withheld_digests", None)
    if not callable(reader):
        return []
    retries = {}
    if ledger is not None:
        with closing(ledger._connect()) as conn:
            retries = {row[0]: (row[1], row[2]) for row in conn.execute(
                "SELECT contact_id, last_attempt, retry_at FROM contact_digest_retries")}
    candidates, offset, page = [], 0, 200
    while True:
        rows = await reader(limit=page, offset=offset)
        for record in rows or []:
            fields = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            cid = str(fields.get("contact_id") or "")
            attempted, retry_at = retries.get(cid, (0, 0))
            if quiet_seconds:
                try:
                    changed = datetime.fromisoformat(fields["digest_withheld_at"].replace("Z", "+00:00"))
                    if now - changed.timestamp() < quiet_seconds:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
            if cid and cid != owner_id and retry_at <= now:
                candidates.append((attempted, str(fields.get("digest_withheld_at") or ""), cid, record))
        if len(rows or []) < page:
            break
        offset += page
    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates[:limit]]
