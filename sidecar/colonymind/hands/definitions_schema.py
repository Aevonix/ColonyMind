"""Forward-only v2 extension of hands.db; the execution schema stays intact."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS hand_definitions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, enabled INTEGER NOT NULL,
    version INTEGER NOT NULL CHECK(version>=1), archived_at TEXT,
    document TEXT NOT NULL CHECK(json_valid(document))
);
CREATE UNIQUE INDEX IF NOT EXISTS hand_definition_name ON hand_definitions(name) WHERE archived_at IS NULL;
CREATE TABLE IF NOT EXISTS hand_definition_versions (
    definition_id TEXT NOT NULL REFERENCES hand_definitions(id), version INTEGER NOT NULL,
    document TEXT NOT NULL CHECK(json_valid(document)), PRIMARY KEY(definition_id,version)
);
CREATE TABLE IF NOT EXISTS task_definitions (
    task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
    definition_id TEXT NOT NULL, version INTEGER NOT NULL,
    snapshot TEXT NOT NULL CHECK(json_valid(snapshot)),
    FOREIGN KEY(definition_id,version) REFERENCES hand_definition_versions(definition_id,version)
);
CREATE INDEX IF NOT EXISTS task_definition_lookup ON task_definitions(definition_id,task_id);
CREATE TABLE IF NOT EXISTS definition_task_queue (
    task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
    brief TEXT NOT NULL, dispatched INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS hands_change_identity (id TEXT PRIMARY KEY);
INSERT INTO hands_change_identity SELECT lower(hex(randomblob(16)))
WHERE NOT EXISTS (SELECT 1 FROM hands_change_identity);
CREATE TABLE IF NOT EXISTS hands_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    definition_id TEXT, task_id TEXT, change TEXT NOT NULL, state TEXT, version INTEGER,
    published INTEGER NOT NULL DEFAULT 0, prepared INTEGER NOT NULL DEFAULT 0
);
CREATE TRIGGER IF NOT EXISTS definition_created AFTER INSERT ON hand_definitions BEGIN
    INSERT INTO hands_changes(definition_id,change,state,version)
    VALUES(NEW.id,'definition.created',CASE WHEN NEW.enabled THEN 'enabled' ELSE 'disabled' END,NEW.version);
END;
CREATE TRIGGER IF NOT EXISTS definition_updated AFTER UPDATE ON hand_definitions BEGIN
    INSERT INTO hands_changes(definition_id,change,state,version)
    VALUES(NEW.id,CASE WHEN NEW.archived_at IS NULL THEN 'definition.updated' ELSE 'definition.archived' END,
           CASE WHEN NEW.archived_at IS NOT NULL THEN 'archived' WHEN NEW.enabled THEN 'enabled' ELSE 'disabled' END,NEW.version);
END;
CREATE TRIGGER IF NOT EXISTS hands_task_created AFTER INSERT ON tasks BEGIN
    INSERT INTO hands_changes(task_id,change,state) VALUES(NEW.id,'task.created',NEW.state);
END;
CREATE TRIGGER IF NOT EXISTS hands_task_changed AFTER UPDATE OF state,intent_version ON tasks
WHEN OLD.state!=NEW.state OR OLD.intent_version!=NEW.intent_version BEGIN
    INSERT INTO hands_changes(task_id,change,state) VALUES(NEW.id,'task.state',NEW.state);
END;
"""

# All runtime state producers participate, including asks, permits and review.
for table, columns, task in (
    ("hands", "state,activity", "NEW.task_id"),
    ("questions", "state,blocking", "(SELECT task_id FROM hands WHERE id=NEW.hand_id)"),
    ("permits", "state", "NEW.task_id"),
    ("obligations", "state,reviewed_at,accepted_at", "NEW.task_id"),
):
    for operation, suffix in (("INSERT", "created"), (f"UPDATE OF {columns}", "changed")):
        condition = ("WHEN " + " OR ".join(f"OLD.{column} IS NOT NEW.{column}" for column in columns.split(','))
                     if suffix == 'changed' else '')
        SCHEMA += f"""
CREATE TRIGGER IF NOT EXISTS hands_{table}_{suffix} AFTER {operation} ON {table} {condition} BEGIN
    INSERT INTO hands_changes(task_id,change) VALUES({task},'task.state');
END;
"""

# Correlation is filled at publish time: task.created precedes its snapshot binding.

QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS definition_task_dispatch (
    task_id TEXT PRIMARY KEY REFERENCES definition_task_queue(task_id) ON DELETE CASCADE,
    claim_token TEXT UNIQUE, lease_until TEXT, parent_session TEXT, hand_id TEXT,
    dispatch_state TEXT NOT NULL DEFAULT 'queued', checked_at TEXT
);
CREATE TRIGGER IF NOT EXISTS definition_dispatch_changed AFTER UPDATE OF dispatch_state ON definition_task_dispatch
WHEN OLD.dispatch_state != NEW.dispatch_state BEGIN
    INSERT INTO hands_changes(task_id,change) VALUES(NEW.task_id,'task.state');
END;
"""

# v3 rows have no verifiable initiating authority. Preserve them for the owner
# to inspect, but never infer an owner start from their body or saved parent.
OWNER_QUEUE_SCHEMA = """
ALTER TABLE definition_task_queue ADD COLUMN initiating_principal TEXT;
ALTER TABLE definition_task_queue ADD COLUMN initiating_credential TEXT;
ALTER TABLE definition_task_queue ADD COLUMN owner_started INTEGER NOT NULL DEFAULT 0 CHECK(owner_started IN (0,1));
INSERT OR IGNORE INTO definition_task_dispatch(task_id)
    SELECT task_id FROM definition_task_queue WHERE dispatched=0;
UPDATE definition_task_dispatch SET dispatch_state='awaiting_owner_start',
    claim_token=NULL,lease_until=NULL,parent_session=NULL
    WHERE task_id IN (SELECT task_id FROM definition_task_queue WHERE dispatched=0);
CREATE INDEX definition_queue_pending ON definition_task_queue(task_id) WHERE dispatched=0;
CREATE INDEX definition_queue_claimable ON definition_task_queue(task_id) WHERE dispatched=0 AND owner_started=1;
CREATE INDEX definition_dispatch_expiry ON definition_task_dispatch(lease_until) WHERE lease_until IS NOT NULL;
CREATE INDEX hands_changes_unprepared ON hands_changes(id) WHERE prepared=0;
CREATE INDEX hands_changes_unpublished ON hands_changes(id) WHERE published=0 AND prepared=1;
"""

# Eligibility is operational state, independent of immutable initiating evidence
# and the consumed-binding fence. Default closed for imported/history rows.
ELIGIBLE_QUEUE_SCHEMA = """
ALTER TABLE definition_task_queue ADD COLUMN eligible INTEGER NOT NULL DEFAULT 0 CHECK(eligible IN (0,1));
UPDATE definition_task_queue SET eligible=1 WHERE dispatched=0 AND owner_started=1
    AND task_id IN (SELECT id FROM tasks WHERE state IN ('open','waiting') AND intent_version=1)
    AND NOT EXISTS (SELECT 1 FROM hands h WHERE h.task_id=definition_task_queue.task_id
        AND h.state IN ('done','below_minimum','failed','stopped','superseded','lost','unknown_effect'));
UPDATE definition_task_dispatch SET claim_token=NULL,lease_until=NULL,parent_session=NULL
    WHERE task_id IN (SELECT task_id FROM definition_task_queue WHERE eligible=0 AND dispatched=0);
DROP INDEX definition_queue_claimable;
CREATE INDEX definition_queue_claimable ON definition_task_queue(task_id)
    WHERE dispatched=0 AND owner_started=1 AND eligible=1;
CREATE TRIGGER definition_queue_task_closed AFTER UPDATE OF state,intent_version ON tasks
WHEN NEW.state NOT IN ('open','waiting') OR OLD.intent_version!=NEW.intent_version BEGIN
    UPDATE definition_task_queue SET eligible=0 WHERE task_id=NEW.id;
    UPDATE definition_task_dispatch SET claim_token=NULL,lease_until=NULL,parent_session=NULL
        WHERE task_id=NEW.id AND EXISTS
            (SELECT 1 FROM definition_task_queue WHERE task_id=NEW.id AND dispatched=0);
END;
-- Refusals and expired leases are retryable admission outcomes. Owner stops,
-- supersession and execution outcomes must never revive the original request.
CREATE TRIGGER definition_queue_attempt_closed AFTER UPDATE OF state ON hands
WHEN NEW.state IN ('done','below_minimum','failed','stopped','superseded','lost','unknown_effect') BEGIN
    UPDATE definition_task_queue SET eligible=0 WHERE task_id=NEW.task_id;
    UPDATE definition_task_dispatch SET claim_token=NULL,lease_until=NULL,parent_session=NULL
        WHERE task_id=NEW.task_id AND EXISTS
            (SELECT 1 FROM definition_task_queue WHERE task_id=NEW.task_id AND dispatched=0);
END;
"""
