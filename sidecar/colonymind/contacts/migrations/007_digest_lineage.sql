-- The per-contact digest's lineage: which writer produced it, beside the claim ids it was built from
-- (digest_sources). 'template' is the tick's rendered digest, 'night' the consolidation's generated one;
-- NULL while there is no digest. digest_withheld_at: when a claim the digest cited stopped being live
-- (erased or retracted); its text is gone and it waits for a rebuild. NULL = nothing withheld.
ALTER TABLE contacts ADD COLUMN digest_writer TEXT CHECK (digest_writer IN ('template','night'));
ALTER TABLE contacts ADD COLUMN digest_withheld_at TEXT;
CREATE INDEX IF NOT EXISTS idx_contacts_digest_withheld
  ON contacts(digest_withheld_at) WHERE digest_withheld_at IS NOT NULL;
UPDATE contacts SET digest_writer = 'night' WHERE digest IS NOT NULL AND digest_sources LIKE '["claim:%';
-- A template digest written before it recorded its claims (sources ["template"]) cites nothing it can be
-- checked against, so its text may carry a claim erased since. It is withheld: the mind's next tick renders
-- it again from the live claims (no model call), with its lineage, whether the mind is on or off (with the
-- mind off the tick still does this upkeep, Mind._digest_upkeep).
UPDATE contacts SET digest_writer = 'template', digest = NULL, digest_sources = '[]',
  digest_withheld_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
  WHERE digest IS NOT NULL AND digest_sources = '["template"]';
