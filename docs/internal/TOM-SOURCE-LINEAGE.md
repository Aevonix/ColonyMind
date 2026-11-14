# Source lineage of the contacts' stores

The comms log (`contacts.comms.CommsLog`) and the contacts' affect
(`contacts.affect.AffectStore`) are projections kept beside the canonical source
ledger. Both extend `contacts.source_lineage.SourceLinkedStore`: a record derived
from a conversation keeps the canonical turn ID, session and exact message hashes
of its support, and only a person-scoped canonical source can support it. A
session checkpoint, or a turn that is absent or erased, links nothing.

The appraisal of a non-owner speaker's turn writes the speaker's apparent valence
into the contacts' affect (`contacts.affect_writer`); the mind's social drive
reads its trend to hold check-ins while a contact's affect declines. Nothing else
writes it: its manual routes were removed in the prune.

Forgetting commits the source tombstone first, then removes the linked rows. The
response reports `affect_cleanup` and `communications_cleanup`; `pending` means
physical cleanup must be retried. Reads check source validity even when cleanup
fails, late writes recheck the source, and repeating the forget operation retries
cleanup. A merge moves a contact's sourced rows once their sources moved.

The shared-facts store is gone: no writer was left (its manual route and the MCP
tool were removed) and automatic recall showed only source-linked facts, so it
contributed nothing. `colonymind upgrade` moves `colonymind-facts.db` into the
upgrade backup; nothing deletes it.

`test_affect.py`, `test_source_appraisals.py` and the erasure tests exercise the
lineage, the appraisal writer and the cleanup on the actual SQLite stores. These
checks establish storage behavior, not model appraisal quality.
