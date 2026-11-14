# Owner goals

Owner goals use the existing intentions database with the kinds `owner_goal`,
`owner_subgoal` and `owner_goal_progress`. Their engine status is `recorded`;
accepted lifecycle status lives in context as `active` or `completed`. They have
no autonomous goal budget, horizon, expiry or success check. Generic initiative
routes and mind audit projections exclude them. Their own audience-bound reader
is the supported projection. Progress is immutable; activity derives from progress
and original run output. Deletion records tombstones and retains audit, progress,
proposals, attachment evidence and receipts. Generic retention excludes these kinds.

## Storage sealing and integrity

The owner store seals content on every write with the existing secrets vault.
The shared detector and JSON leaf traversal identify credentials; the complete
containing text field is encrypted by `Vault.seal`, preserving its exact bytes.
This includes goal context, progress, proposals, attachment labels and bindings,
audit details, receipt results and resolved actions. The default detector rules
apply even when conversational capture is disabled. Relationship identities,
authority identities and integrity digests remain structural. Approved owner
reads decode the storage representation; arbitrary secret references in original
content are never expanded. Every owner write connection, including schema and
sealing migrations, sets `PRAGMA secure_delete=ON` before writing.

An owner-store metadata table stores sensitivity, inheritance override, tombstone,
status and progress ordering fields without content. Writes update it in the same
transaction as the record. Readers resolve audience eligibility and inherited
tiers from this metadata before decrypting any record, including collection
sorting, previews and activity. Missing legacy metadata denies non-owner access
until migration backfills it. Hidden content is never decrypted and reads exactly
like an absent row, even when its vault entry is forgotten or unavailable.
Permitted reads with unavailable ciphertext return HTTP 503 with
`owner_goal_content_unavailable`; they never return a decryption HTTP 500.

The secrets migration calls `OwnerGoalsStore.seal_existing` explicitly. Generic
sealing and lineage walkers continue to exclude every `owner_goal_*` table from
reads and writes. The repair uses `migration:v3-owner-goals-physical`; an earlier
completion marker cannot skip it. Existing content, including tombstones and legacy engine
metadata, is sealed one row per transaction. The vault commits ciphertext before
the row references it. An interruption can leave an unused encrypted vault entry;
the retry reuses that entry and skips already sealed fields. A verification pass
and physical scrub finish before completion is recorded. `VACUUM` rebuilds the
database to remove old cells and free pages left by earlier writers, including
residue that `secure_delete` cannot retroactively erase. In WAL mode the rebuild
serializes writers while existing readers may retain their snapshots. Budget up
to twice the database size in additional disk space for SQLite's rebuild, plus
WAL growth. The following `wal_checkpoint(TRUNCATE)` needs readers to release
their snapshots; connections wait up to 30 seconds for locks. Busy readers, lock
contention, insufficient disk space or interruptions fail the pass and leave it
resumable without recording completion. The migration never removes sidecar files
or replaces the database behind live connections.

After the scrub and checkpoint, a write reservation prevents concurrent writes
while raw database and WAL bytes are scanned for credential canaries collected
from the original or decrypted owner content (including JSON-escaped forms).
Canaries stay in memory and never appear in reports. Any surviving bytes,
remaining logical plaintext, unavailable ciphertext or blocked checkpoint prevents
completion. A dry run only counts fields and does not create a vault or change schema.

The parent commit's generic sealing also left physical plaintext residue from
older cells/free pages despite logically sealed rows and `secure_delete` during
migration. This is a pre-existing generic issue. Generic sealing remains outside
this owner-store repair; the physical scrub and verification are the explicit
owner-store migration's responsibility.

Progress, audit and receipt immutability covers logical bytes. Their update
triggers remain active during sealing: only the sealing connection registers a
callback that accepts an encoding change when the new ciphertext decrypts to the
identical original value and every other column is unchanged. Ordinary owner
connections reject those updates; other connections lack the callback and fail
closed. Deletes remain forbidden. Each row update and its constraints commit or
roll back together.

Receipt `request_hash` retains SHA-256 over the canonical serialization of
`[method, path, body]`. On replay the store recomputes this digest from the retry's
original body, method and canonical path and compares it with the retained hash.
The resolved action decrypts to the original validated effects, including defaults,
so the verifier can recheck those effects and the retained evidence binding.
Proposal audit `payload_digest`
retains SHA-256 over the canonical original payload. Decoding the proposal and
audit details permits the same check after sealing. No audit event, revision,
timestamp, request hash or payload digest is rewritten. Receipt replay decodes
the original result and resolved action before verification and returns the
original status and result.

The router is mounted by `mind_routers()` at `/v1/mind/owner-goals`. Requests are
parsed strictly, including duplicate keys, Unicode scalars, finite numbers,
source identifiers, safe integers, timestamps and unknown fields. Collection
queries accept only single decimal `limit` and `offset` values. Pages and nested
previews carry integer continuation offsets, without a history cap. Projections
shrink pages to the response byte budget and never produce empty nonterminal pages.

## Owner gate integration

`get_action_verifier` is the injected dependency for every mutation. Its default
`verify_owner_goal_action` denies execution and replay: missing source or action
capability returns `owner_turn_required`; an unrecognized capability returns
`not_owner`. No JSON identity, sender, approval, mode or email field grants
permission. The owner gate must implement the `ActionVerifier` protocol in
`colonymind.owner_goals.authority` and provide a trusted `Audience` through
`request.state.owner_goal_audience`. GET refuses unavailable audience bindings;
an owner credential alone cannot establish a conversation audience.

The verifier runs exactly once for an admitted mutation or matching receipt,
inside the same SQLite write transaction. `ActionContext` contains the principal,
server owner, complete request body/hash, method, canonical path, operation UUID
and request transport. `resolved_action` includes defaults, target revisions,
proposal payload/version and attachment authority bindings. The owner gate must
check current authenticated ingress/chat/audience, source/version, live action
capability and exact effects there. It returns a server-derived `VerifiedAction`.
Tier overrides require explicit tier authority. Proposal submission uses
`proposal_only`, accepted writes use `requested_write`, and exact confirmation uses
`confirmation`. The default grants no phone exemption; only the owner gate may
recognize an isolated relay through authenticated server context.

Receipt lookup is scoped to principal, owner and operation UUID. Changed requests
conflict before dispatch. Matching receipts are verified against their retained
evidence binding and replayed before target/revision/settlement lookups. The owner
gate must authorize retries against the current trusted chat and audience; a bare
bearer or expired execution capability cannot establish that binding. Mutation,
audit, proposal settlement and the original status/result commit together. Failures
while building required projections roll the entire transaction back.

Secret is the default. Child goals and progress inherit their current parent's
tier unless an explicit owner-authorized override was recorded. Inherited tier
changes update projections and audit without rewriting original progress. Child
revisions advance when inherited tier projections change. Proposal payloads stay
immutable while their read sensitivity follows inherited target tiers. Secret is
excluded from mixed and non-owner recall, even with an owner credential. Reads
never confer permission to share content. Outreach eligibility also checks every
ancestor's mute, completion and tombstone state.

## Artifact integration

`ArtifactResolver` implements `resolve_owner_artifact` using a configured
`LibraryReader` and `CronReader`. Library records must supply canonical namespace,
object version, authoritative owner association or owner ACL, ACL version, content
tier and any stricter reader ACL. Public visibility and an ID are insufficient.
Every relation read revalidates current authority and pins the original namespace.
Known missing/foreign attach targets return `artifact_not_found`; absent authority
returns `owner_goals_unavailable`. Lost eligibility on an existing relation is
unavailable. Content excluded only by the current audience is filtered before
pagination, including metadata and activity.

This repository has no existing Library ownership/ACL service. `LibraryReader`
is the explicit integration boundary; its default remains unavailable. The cron
boundary similarly requires the read adapter to verify profile/job destination
ownership and historical per-run delivery evidence. Original run pages are merged
with progress in timestamp/source order, with offsets applied to the merged
stream. Invalid or incomplete output pages fail unavailable. Attachment operations
never control a scheduler; detach and tombstone need no removed job output.
Migration enumeration, delivery proof acquisition and scheduler controls belong
to their separate adapters, not this store or its public routes.
