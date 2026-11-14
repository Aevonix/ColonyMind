"""Owner-owned storage encoding; logical content and its hashes never change."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from colonymind.secrets.detect import find_secrets
from colonymind.secrets.seal import map_leaves
from colonymind.secrets.vault import VaultUnavailable, get_vault

# A control character cannot arrive through the owner request's text validation.
PREFIX = "\x1eowner-goal:v1:"
# Only relationship/authority identities and integrity digests stay structural.
# All other text, including retained legacy engine metadata, is content.
STRUCTURAL = {
    "owner_goal_records": {"id", "kind", "entity_id", "parent_goal_id"},
    "owner_goal_history": {"id", "initiative_id", "agent_id"},
    "owner_goal_proposals": {"id", "owner_id", "target_id"},
    "owner_goal_artifacts": {"goal_id", "type", "ref"},
    "owner_goal_receipts": {"principal_id", "owner_id", "operation_id", "request_hash", "evidence_id"},
    "owner_goal_legacy_answers": {"scope", "key", "ask_id", "request_hash"},
    "owner_goal_legacy_acceptances": {"acceptance_key", "initiative_id", "handoff_hash"},
}


def sensitive(value):
    """Use the shared detector and JSON field/leaf traversal, including named values."""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    def mark(text):
        return text + "\x00" if find_secrets(text) else text
    return map_leaves(parsed, mark) != parsed


def plaintext_canaries(value):
    """Known credential bytes, including JSON escapes; held in memory only."""
    if not isinstance(value, str):
        return set()
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    canaries = set()
    def collect(text):
        for hit in find_secrets(text):
            for form in (hit.value, json.dumps(hit.value, ensure_ascii=True)[1:-1],
                         json.dumps(hit.value, ensure_ascii=False)[1:-1]):
                for encoding in ("utf-8", "utf-16le", "utf-16be"):
                    canaries.add(form.encode(encoding))
        return text
    map_leaves(parsed, collect)
    return canaries


def verify_physical_sealing(path, canaries):
    """Scan the database and WAL under the caller's writer lock, including old cells."""
    overlap = max(map(len, canaries), default=1) - 1
    for candidate in (path, Path(str(path) + "-wal")):
        if not candidate.exists():
            continue
        with candidate.open("rb") as stream:
            tail = b""
            while chunk := stream.read(1024 * 1024):
                block = tail + chunk
                if any(canary in block for canary in canaries):
                    raise VaultUnavailable("owner-goal plaintext remains in database storage")
                tail = block[-overlap:] if overlap else b""


class Codec:
    def __init__(self, state_dir, vault=None):
        self.state_dir, self._vault = Path(state_dir), vault

    @property
    def vault(self):
        if self._vault is None:
            from colonymind.secrets.config import current_settings
            self._vault = get_vault(self.state_dir, key_store=current_settings().key_store)
        return self._vault

    def plaintext(self, value):
        if not isinstance(value, str) or not value.startswith(PREFIX):
            return value
        try:
            return self.vault.resolve(value[len(PREFIX):])
        except Exception:
            raise VaultUnavailable("owner-goal content cannot be opened") from None

    def seal(self, value, owner):
        # Inputs are logical values, including strings that happen to name this encoding.
        if not isinstance(value, str) or not (sensitive(value) or value.startswith(PREFIX)):
            return value
        ref = self.vault.seal(value, contact_id=owner, kind="owner_goal_content", label="owner goal content")
        if ref is None:
            raise VaultUnavailable("owner-goal content could not be sealed")
        return PREFIX + ref

    def equivalent(self, before, after):
        """Only an encoding change to identical bytes may cross an immutable-row trigger."""
        old, new = json.loads(before), json.loads(after)
        return len(old) == len(new) and all(
            a == b or (isinstance(b, str) and b.startswith(PREFIX) and self.plaintext(b) == a)
            for a, b in zip(old, new)
        )


def seal_existing(path, *, vault=None, dry_run=False):
    """Commit one equivalent row encoding at a time; retries skip committed ciphertext.

    This is the explicit migration hook. Generic walkers never open owner tables.
    Failures propagate, so the caller cannot mark the migration complete.
    """
    from .schema import initialize_sealing, initialize_visibility, save_visibility
    from .store import OwnerGoalsStore
    path = Path(path)
    if not path.is_file():
        return {}
    with closing(sqlite3.connect(path, timeout=30)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='owner_goal_records'").fetchone():
            return {}
        if not dry_run:
            with db:
                db.execute("PRAGMA secure_delete=ON")
                db.execute("BEGIN IMMEDIATE")
                initialize_sealing(db)
                initialize_visibility(db)
    store = OwnerGoalsStore(path, vault=vault)
    counts = {}
    canaries = set()
    for table, structural in STRUCTURAL.items():
        with store.connection() as db:
            content = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')
                       if row[1] not in structural and "TEXT" in row[2].upper()]
        cursor = None
        while True:
            with store.connection(write=not dry_run) as db:
                where, args = ("", ()) if cursor is None else ("WHERE rowid>?", (cursor,))
                row = db.execute(f'SELECT rowid AS _rid, * FROM "{table}" {where} '
                                 'ORDER BY rowid LIMIT 1', args).fetchone()
                if row is None:
                    break
                cursor = row["_rid"]
                owner = row["entity_id"] if table == "owner_goal_records" else (
                    row["owner_id"] if "owner_id" in row.keys() else "unscoped")
                target = next((row[c] for c in ("initiative_id", "goal_id", "ask_id") if c in row.keys()), None)
                if target is not None:
                    parent = db.execute("""SELECT entity_id FROM owner_goal_records WHERE id=?
                        UNION ALL SELECT owner_id FROM owner_goal_proposals WHERE id=?""", (target, target)).fetchone()
                    if parent is not None:
                        owner = parent[0]
                updates = {}
                for column in content:
                    value = row[column]
                    if isinstance(value, str) and value.startswith(PREFIX):
                        # Verify existing ciphertext too; an unavailable vault cannot certify completion.
                        if not dry_run:
                            canaries.update(plaintext_canaries(store.codec.plaintext(value)))
                    elif sensitive(value):
                        if not dry_run:
                            canaries.update(plaintext_canaries(value))
                        updates[column] = value if dry_run else store.codec.seal(value, owner)
                        if not dry_run and store.codec.plaintext(updates[column]) != value:
                            raise VaultUnavailable("owner-goal sealing changed logical content")
                        counts[(path.name, table, column)] = counts.get((path.name, table, column), 0) + 1
                if table == "owner_goal_records" and not dry_run:
                    save_visibility(db, json.loads(store.codec.plaintext(row["context"])))
                if updates and not dry_run:
                    # Only this connection, while holding the write lock, can perform an
                    # equivalent representation change. No immutable trigger is disabled.
                    db.create_function("owner_goal_storage_rewrite", 2, store.codec.equivalent)
                    db.execute(f'UPDATE "{table}" SET ' + ','.join(f'"{c}"=?' for c in updates)
                               + ' WHERE rowid=?', (*updates.values(), cursor))
    if not dry_run:
        if seal_existing(path, vault=store.codec.vault, dry_run=True):
            raise VaultUnavailable("owner-goal content remains unsealed")
        # VACUUM scrubs fragments left by earlier writers with secure_delete disabled.
        # In WAL mode it serializes writers; readers may continue on their snapshots.
        # It needs temporary disk space up to twice the database size. Checkpointing
        # then requires those readers to release their snapshots (30s busy timeout).
        with closing(sqlite3.connect(path, timeout=30)) as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("VACUUM")
            if db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                raise VaultUnavailable("owner-goal sealing awaits active readers")
            # Prevent a concurrent writer from changing either file during the scan.
            db.execute("BEGIN IMMEDIATE")
            try:
                verify_physical_sealing(path, canaries)
            finally:
                db.rollback()
    return counts
