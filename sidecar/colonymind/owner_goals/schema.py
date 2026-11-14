"""Owner-goals tables and the transactional, one-time shared-table migration."""
from __future__ import annotations

import sqlite3
import re


OWNER_KINDS_SQL = "('owner_goal','owner_subgoal','owner_goal_progress')"


def initialize_visibility(db):
    """Keep only access-control and ordering metadata outside encrypted content.

    Already encrypted legacy rows are backfilled by the explicit sealing hook.
    Until then missing metadata denies non-owner reads without opening the vault.
    """
    if db.execute("SELECT 1 FROM owner_goal_schema WHERE version=3").fetchone():
        return
    db.execute("""CREATE TABLE IF NOT EXISTS owner_goal_visibility (
        record_id TEXT PRIMARY KEY, sensitivity TEXT NOT NULL
            CHECK(sensitivity IN ('Public','Everyday','Personal','Private','Secret')),
        tier_override INTEGER NOT NULL CHECK(tier_override IN (0,1)),
        deleted INTEGER NOT NULL CHECK(deleted IN (0,1)),
        status TEXT, sort_at TEXT)""")
    db.execute("""INSERT OR IGNORE INTO owner_goal_visibility
        SELECT id,json_extract(context,'$.sensitivity'),
               COALESCE(json_extract(context,'$._tier_override'),0),
               json_extract(context,'$._deleted_at') IS NOT NULL,
               json_extract(context,'$.status'),json_extract(context,'$._sort_at')
        FROM owner_goal_records WHERE CASE WHEN json_valid(context)
            THEN json_extract(context,'$.sensitivity') IN ('Public','Everyday','Personal','Private','Secret')
            ELSE 0 END""")
    db.execute("INSERT INTO owner_goal_schema VALUES (3)")


def save_visibility(db, data):
    db.execute("""INSERT INTO owner_goal_visibility VALUES (?,?,?,?,?,?)
        ON CONFLICT(record_id) DO UPDATE SET sensitivity=excluded.sensitivity,
        tier_override=excluded.tier_override,deleted=excluded.deleted,
        status=excluded.status,sort_at=excluded.sort_at""",
        (data["id"], data["sensitivity"], int(bool(data.get("_tier_override"))),
         int(bool(data.get("_deleted_at"))), data.get("status"), data.get("_sort_at")))


def initialize_sealing(db):
    """Add a fail-closed, owner-store-only equivalent storage rewrite to frozen rows."""
    if db.execute("SELECT 1 FROM owner_goal_schema WHERE version=2").fetchone():
        return
    for table, trigger, condition, error in (
        ("owner_goal_records", "owner_goal_progress_immutable", "OLD.kind='owner_goal_progress' AND ",
         "owner_goal_progress_immutable"),
        ("owner_goal_history", "owner_goal_audit_no_update", "", "owner_goal_audit_immutable"),
        ("owner_goal_receipts", "owner_goal_receipts_no_update", "", "owner_goal_receipts_immutable"),
    ):
        columns = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
        old = ','.join(f'OLD."{column}"' for column in columns)
        new = ','.join(f'NEW."{column}"' for column in columns)
        db.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        db.execute(f'''CREATE TRIGGER "{trigger}" BEFORE UPDATE ON "{table}"
            WHEN {condition} NOT owner_goal_storage_rewrite(json_array({old}),json_array({new}))
            BEGIN SELECT RAISE(ABORT,'{error}'); END''')
    db.execute("""CREATE TRIGGER owner_goal_receipts_no_delete BEFORE DELETE ON owner_goal_receipts
        BEGIN SELECT RAISE(ABORT,'owner_goal_receipts_immutable'); END""")
    db.execute("INSERT INTO owner_goal_schema VALUES (2)")


def _copy_schema(db, source, target, *, references=None):
    # Keep every legacy column, default and constraint, including engine metadata
    # retained for audit. These names are internal constants, never request input.
    sql = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (source,)).fetchone()[0]
    definition = sql[sql.index("("):]
    if references:
        # Audit targets may be either records or proposals, so the legacy
        # single-table foreign key cannot represent this owner-only relation.
        definition = re.sub(r",\s*FOREIGN KEY\s*\(initiative_id\)\s*REFERENCES initiatives\s*\(id\)",
                            "", definition, flags=re.I)
    if target == "owner_goal_records":
        definition = definition.rstrip().removesuffix(")") + f", CHECK(kind IS NOT NULL AND kind IN {OWNER_KINDS_SQL}))"
    db.execute(f"CREATE TABLE IF NOT EXISTS {target} {definition}")


def initialize(db: sqlite3.Connection) -> None:
    """Run before exposing InitiativeStore, including when no owner API is loaded.

    The marker, copies, removal, and replacement constraints commit together.
    A collision fails the whole transaction; no retained data is overwritten.
    Subsequent opens check the primary-key marker without scanning either store.
    """
    db.execute("PRAGMA secure_delete=ON")
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS owner_goal_schema (version INTEGER PRIMARY KEY)")
        if db.execute("SELECT 1 FROM owner_goal_schema WHERE version=1").fetchone():
            initialize_sealing(db)
            initialize_visibility(db)
            db.commit()
            return
        _copy_schema(db, "initiatives", "owner_goal_records")
        _copy_schema(db, "assignment_history", "owner_goal_history", references=True)
        _copy_schema(db, "ask_answer_requests", "owner_goal_legacy_answers")
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='local_draft_acceptances'").fetchone():
            _copy_schema(db, "local_draft_acceptances", "owner_goal_legacy_acceptances")
            acceptances = True
        else:
            db.execute("""CREATE TABLE owner_goal_legacy_acceptances (
                acceptance_key TEXT PRIMARY KEY, initiative_id TEXT NOT NULL,
                handoff_hash TEXT NOT NULL DEFAULT '')""")
            acceptances = False
        db.execute("""CREATE TABLE IF NOT EXISTS owner_goal_proposals (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, data TEXT NOT NULL,
            tier_override INTEGER NOT NULL, target_id TEXT, created_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS owner_goal_artifacts (
            goal_id TEXT NOT NULL, type TEXT NOT NULL, ref TEXT NOT NULL,
            label TEXT NOT NULL, added_at TEXT NOT NULL, binding TEXT NOT NULL,
            detached_at TEXT, PRIMARY KEY(goal_id,type,ref))""")
        db.execute("""CREATE TABLE IF NOT EXISTS owner_goal_receipts (
            principal_id TEXT NOT NULL, owner_id TEXT NOT NULL, operation_id TEXT NOT NULL,
            request_hash TEXT NOT NULL, status INTEGER NOT NULL, result TEXT NOT NULL,
            evidence_id TEXT NOT NULL, resolved_action TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(principal_id,owner_id,operation_id))""")
        owner_ids = f"SELECT id FROM initiatives WHERE kind IN {OWNER_KINDS_SQL}"
        owner_targets = f"{owner_ids} UNION SELECT id FROM owner_goal_proposals"
        history_scope = f"initiative_id IN ({owner_targets}) OR action GLOB 'owner_goal_*'"
        db.execute(f"INSERT INTO owner_goal_records SELECT * FROM initiatives WHERE kind IN {OWNER_KINDS_SQL}")
        db.execute(f"INSERT INTO owner_goal_history SELECT * FROM assignment_history WHERE {history_scope}")
        db.execute(f"INSERT INTO owner_goal_legacy_answers SELECT * FROM ask_answer_requests WHERE ask_id IN ({owner_targets})")
        if acceptances:
            db.execute(f"INSERT INTO owner_goal_legacy_acceptances SELECT * FROM local_draft_acceptances WHERE initiative_id IN ({owner_targets})")
        # Replace the old shared-table retention triggers inside this transaction.
        # The copied rows acquire the same protections before commit.
        for trigger in ("owner_goal_progress_immutable", "owner_goal_records_retained",
                        "owner_goal_audit_no_update", "owner_goal_audit_no_delete"):
            db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        db.execute(f"DELETE FROM assignment_history WHERE {history_scope}")
        db.execute(f"DELETE FROM ask_answer_requests WHERE ask_id IN ({owner_targets})")
        if acceptances:
            db.execute(f"DELETE FROM local_draft_acceptances WHERE initiative_id IN ({owner_targets})")
        db.execute(f"DELETE FROM initiatives WHERE kind IN {OWNER_KINDS_SQL}")
        db.execute("DROP INDEX IF EXISTS owner_goal_children")
        db.execute("CREATE INDEX owner_goal_children ON owner_goal_records(kind,entity_id,parent_goal_id)")
        db.execute("CREATE INDEX IF NOT EXISTS owner_goal_history_record ON owner_goal_history(initiative_id)")
        db.execute("CREATE INDEX IF NOT EXISTS owner_goal_proposal_owner ON owner_goal_proposals(owner_id,created_at,id)")
        db.execute("""CREATE TRIGGER owner_goal_progress_immutable
            BEFORE UPDATE ON owner_goal_records WHEN OLD.kind='owner_goal_progress'
            BEGIN SELECT RAISE(ABORT,'owner_goal_progress_immutable'); END""")
        db.execute("""CREATE TRIGGER owner_goal_records_retained BEFORE DELETE ON owner_goal_records
            BEGIN SELECT RAISE(ABORT,'owner_goal_records_retained'); END""")
        for verb in ("UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER owner_goal_audit_no_{verb.lower()}
                BEFORE {verb} ON owner_goal_history
                BEGIN SELECT RAISE(ABORT,'owner_goal_audit_immutable'); END""")
        # Enforce the boundary for direct SQL writers as well as store methods.
        # NEW.kind is row-local; these assertions never inspect another row.
        for verb in ("INSERT", "UPDATE OF kind"):
            name = "insert" if verb == "INSERT" else "update"
            db.execute(f"""CREATE TRIGGER initiatives_reject_owner_{name}
                BEFORE {verb} ON initiatives WHEN NEW.kind IN {OWNER_KINDS_SQL}
                BEGIN SELECT RAISE(ABORT,'Owner kinds require the owner-goals store'); END""")
        db.execute("INSERT INTO owner_goal_schema VALUES (1)")
        initialize_sealing(db)
        initialize_visibility(db)
        db.commit()
    except BaseException:
        db.rollback()
        raise
