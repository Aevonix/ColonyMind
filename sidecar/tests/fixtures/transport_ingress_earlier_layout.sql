-- The receipt table from an earlier layout remains usable after rollback.
CREATE TABLE IF NOT EXISTS transport_ingress (
    receipt_id TEXT PRIMARY KEY, producer TEXT NOT NULL,
    account_id TEXT NOT NULL, epoch TEXT NOT NULL, sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL, contact_id TEXT, occurred_at REAL NOT NULL,
    journal_ref TEXT NOT NULL, payload_digest TEXT NOT NULL,
    media_available INTEGER NOT NULL, metadata_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'admitted', batch_id TEXT,
    native_turn_json TEXT, source_versions_json TEXT,
    outcome TEXT, created_at REAL NOT NULL, completed_at REAL,
    UNIQUE(producer,account_id,epoch,sequence));
