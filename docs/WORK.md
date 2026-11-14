# Verified work

Work records answer what an owner request caused and what the evidence proves.
An assistant saying "done" is never evidence of completion. This contract covers
capture, verification, nightly titles and owner verdict linking.

## Authority and origin

A root belongs to an admitted owner request. The sidecar resolves the sender to
the owner and checks that the calling credential grants that owner platform.
Capture hooks cannot resolve or create contacts. Automated senderless lanes,
cron sessions, background review and kanban workers cannot create roots. CLI
requests are eligible only when `cli` is an owner platform on the credential.
An adapter's candidate flag or claimed contact is not authority.

The root ID is the admitted turn ID. Its source ID, source revision, contact,
session and principal establish its lineage. A gateway message key is retained
as `origin_ref` when available. Hands tasks bind through their existing
`origin=owner_turn` and `origin_ref`, never through similarity of their text.
Subagents bind through the actual `subagent_start.parent_turn_id`, including
nested children. Sharing a session does not establish a parent.

The public API returns a separate opaque `id`, encoded as `w:` followed by
unpadded URL-safe base64 of the original UTF-8 turn ID. Clients use that returned
ID for detail reads. The original `turn_id` and transport lineage stay unchanged.
This makes slashes and reserved route names safe to address.

A successful delegation is `dispatched`. A child completing after the parent
reply produces an independently identified event attached to the original root.
A child success claim does not erase a failed parent operation; only the verified
same-tool, same-resource retry rule below can resolve it. Child effects and hands
outcomes retain their own operation IDs and source dependencies.

## Frozen transport

`schema/work-envelope.json` defines version 1. A turn's optional `work` section
is complete before the existing durable TurnOutbox first enqueues its envelope.
The first envelope wins locally. A retry cannot append or replace operations.
Both `POST /v1/host/turns/sync` and `PUT /v2/host/turns/{id}` carry the same work
section. An absent or null section is removed from canonical hashing, including
session derivation for a request without a session, so old persisted envelopes
keep their original digest and replay response.

When work is supplied, `work_recorded: true` acknowledges source admission even
if projection repair is still pending. An explicit false acknowledgment names
disabled capture or an ineligible origin. A response without that acknowledgment
is an unsupported receiver; the adapter preserves the refused outbox row.

Late events use `POST /v1/host/work/events` and `schema/work-event.json`.
An event ID identifies one immutable observation. An identical retry is accepted;
a changed event or operation with the same identity conflicts. An event cannot
create its own owner root. The root must already have admitted provenance.
Outbox delivery retries a missing parent for at most five minutes after that
event first reaches the sidecar with missing lineage. The first-seen timestamp
is durable and retries do not reset it, even for an old root. Ineligible or
erased roots and expired parent waits return terminal
HTTP 410 (`work_root_ineligible`, `work_root_erased`, `work_parent_expired`).
An unknown root can briefly retry while its turn is still pending. Unsupported work delivery remains
visible as a refused row; downgrade must never silently claim projected work.
Version 1 turn delivery remains the default. A queued `transport_version=2`
selects the second turn route without changing the work envelope.

The ledger admits safe work facts with the source before projection. It is the
durable source of repair after interruption between admission and projection.
Ordinary turn ingestion is not repeated to repair work. Legacy envelopes and
hosts that send no work section remain valid.

## Operations and resources

Every operation names its tool call, tool, action, effect and exact resource.
Resource references are opaque `resource:` slots scoped to a root. Literal paths
and URLs are rejected by the transport schema. A slot is assigned by exact
resource equality in bounded transient
host state, never by hashing raw arguments. Different paths, jobs and message
destinations remain different resources. An unavailable identity remains unknown.

| Tool or action | Classification |
| --- | --- |
| `terminal` explicit read | `read`, only with a known read action |
| `terminal` ordinary execution | `execute` |
| `terminal` background execution | `background` |
| `send_message` list | `read` |
| `send_message` send | `send` |
| `cronjob` list | `read` |
| `cronjob` create, update/edit, remove | `create`, `edit`, `remove` |
| `cronjob` pause, resume, resnap | `edit` |
| `cronjob` run | `dispatch` |
| `write_file`, `patch` | `write`; patch deletion retains `remove` |
| delegation | `dispatch` |
| unrecognized operation | `effect_unknown` |

A tool name pattern, including an MCP name, cannot certify read-only behavior.
Reads can observe an earlier operation only on its exact resource and after it.
A read of another target or an interim assistant claim proves nothing about it.
Multi-resource operations are represented separately per resource. Incomplete
capture adds unknown coverage and can never yield a proven root.

## Evidence and verification

Work stores selected safe facts and references. It does not copy raw tool
arguments, results, command lines, messages, file contents or transcripts.
Historical tool observations remain in `turn_sources.messages_json`, with source lineage
and erasure handling intact. They no longer supply work verification evidence. The
executions API reads selected cron, kanban, reported-worker and Hermes background
process records; it omits local work.

The owner-only turn-start work section and `GET /v1/host/executions` expose
Hermes background processes under `native_processes`. The turn-start list and
job watch use the same read-only Hermes records, selected by `watch.hermes_home`
(default `~/.hermes`): `processes.json` and retained process completion receipts.
The list includes running processes and failures from the last hour, bounded by
the requested limit. Each process line gives the program name, state and elapsed
time since starting or ending; command arguments and output are never included.
Missing, unreadable or timed-out sources return an empty list and unavailable
read coverage. The process read has a 150 ms budget alongside the other readers.

| Operation status | Meaning |
| --- | --- |
| `verified` | The specific postcondition below passed. |
| `observed` | A later read of the exact resource succeeded without checking intent. |
| `ran` | The tool returned success without postcondition proof. |
| `failed` | The call failed or a specific postcondition contradicted it. |
| `dispatched` | Work was delegated and its outcome is pending. |
| `unknown` | The effect or outcome is unavailable. |

Version 1 supports these verifiers:

* `write_file` and `patch`: intended and actual sealed-content SHA256 values
  match. The known host file tool's native post-write SHA256 comparison is also
  accepted as a typed `native_post_write_sha256` attestation. Its documented
  result reports the comparison without exposing raw file hashes. A generic
  `verified` field on another tool is not such an attestation.
* Message sends: the platform delivery receipt or provider message ID is present
  for that successful operation. This proves sent, never read.
* Job create or edit: a later saved-job read for the exact job matches every
  requested field. A create response alone is not a saved-job read.
* Hands: the existing task's verified outcome, joined through `origin_ref`.
  Dispatch or a child's assertion is not a verified hands outcome.

Unknown operations cannot acquire proof from name patterns. An unresolved
failure always wins root aggregation. Without failures, every effect verified
means `proven`; some verified means `partial`; otherwise the root is `unproven`.
Missing coverage is unproven. Pure chat and read/probe-only turns have no
projected root, so they do not enter work counts or digest titles. Their frozen
lineage may still admit a later effect, which then creates a root.
A later successful operation on another target cannot clear a failure.

Reads and probes never count toward root failure (an aggregation assumption). A
later verified operation of the same tool on the same exact resource clears an
earlier failed operation for root aggregation. Any remaining failed effectful
operation makes the root failed. Historical operation statuses remain visible.
An arbitrary terminal execution can be effectful; only classified reads/probes
receive this exception. The owner should confirm this aggregation policy.

The adapter compares job fields in memory, including scalar types, and records
only comparison booleans under `native_saved_job_fields`. A later check after
the root froze has its own operation ID and names `checked_operation_id`.
It cannot rewrite the admitted operation or check a different resource. Stock
patch results without a hash attestation remain `ran`. A formatted job preview
that omits a requested field also leaves the operation unverified.
Display-only fields likewise remain uncheckable rather than causing failure.
Only an explicit false comparison contradicts a requested saved-job field.

A delegation remains dispatched until evidence explicitly resolves that
operation. Verified child effects alone can make its root partial, but cannot
claim that every delegated obligation was fulfilled.

## Safe references and sealing

Sidecar admission uses the existing secrets sealing before summaries, content
digests, work identity digests or persistence. Only sealed representations may
be hashed. A short secret's raw hash is itself unsafe. Supplied file hashes are
allowed only as sealed-content hashes; the adapter uses native attestations
instead of persisting raw-content hashes.

Host work extraction keeps only allowlisted evidence and opaque resource slots.
Selected textual identifiers are passed through existing secret detection and
withholding before durable enqueue; sidecar sealing is applied again before
admission. Detection failures must not fall back to saving raw values. Paths
are lexically normalized in transient adapter state without reading files and
replaced completely by opaque slots before persistence. No home name, path,
URL credential, query or fragment is retained as a resource display label.
No resource identity is inferred from a redacted basename or URL host. Other
hosts must assign their own opaque slots using the same exact-resource rule.

## Concurrency, bounds and lifetime

Work hooks are plain synchronous functions accepting `**kwargs`. They perform
bounded in-memory capture and enqueue work, with no HTTP, model calls, contact
creation or postcondition I/O. Added metadata hooks target p99 below 1 ms. The
existing durable turn enqueue commits before returning and has separate disk
latency; tests report that boundary honestly.

Context is keyed by stable turn ID with a lock, not by the latest session.
At most 256 active contexts and 256 operations per envelope are retained.
Evidence has bounded strings and field counts. A context limit or dropped
operation marks coverage incomplete. Completed state is cleaned up, with
bounded parent bindings retained for late children. An unbound late event is
never attached to the latest owner turn.

Overlapping turns cannot exchange sender, effects or parent IDs. Session
rotation follows the turn ID. Compression checkpoints cannot create a second
root or certify a summary. Interim replies do not freeze a completed record.
Interrupted turns retain observed facts, without inventing completion. Restart
replays frozen outbox rows; it does not reconstruct proof from assistant text.
Durability starts when the root or late event reaches TurnOutbox. A process
death before that boundary can lose its in-memory observations; retained native
evidence remains available for projection repair.

## Projection and erasure

Current source forgetting records curated tombstones and receipts. It suppresses recalled evidence through
the shared memory fence and retains work envelopes, projections and their inspection views. The physical
erasure behavior described below applies to historical erasure records; source forgetting does not invoke it.

`colonymind-work.db` is a repairable projection, not an independent evidence
store. Its roots, children, effects, verification and derived fields rebuild
from admitted ledger facts and hands records. Admission atomically marks only its root dirty. A worker with
its own connection reads that root's source snapshot and writes its projection;
its only canonical-ledger write is a short generation acknowledgment. Turn and
replay handlers wake this worker. Late events mark only their own root. API reads
read the projection without reserving the ledger write lock or rebuilding it.
A dirty generation fences stale rows from reads, including across erasure races.
Cold API reads also bypass ledger and projection initialization: both databases
open read-only. An enabled faculty with no projection yet returns empty lists
and summaries until admission or maintenance creates the projection.

Startup resumes pending generations. An explicit maintenance call,
`python -m colonymind.work` with the configured state directory, rebuilds all roots
and repairs a lost projection. Full repair never runs on a request path. Native
observation lookups use a contact/tool-call index; execution children use turn
and parent-execution indexes. No root scans all owner history or executions.

Owner overlays and titles have separate storage and source lineage.
Execution status never incorporates an assistant title or owner overlay. A
derived write must compare its source revision or generation before committing.
The revision covers only that root's frozen envelopes and retained evidence;
other owner erasures, temporary hands-store failures and expiring execution
observations cannot clear its titles or overlays.

Erasure follows source lineage, including individual messages removed from a
partially retained source. Changed or missing dependencies invalidate affected
effects, titles, overlays and summaries. Erased verdict sources invalidate their
derived overlays. Concurrent stale derived writes are refused. Tombstones fence
replays and rebuilds, including after restart. Erasure must not remove unrelated
roots or silently reattach evidence to a surviving message.

Before replacing a projection, a worker compares its captured generation with a
fresh ledger read while holding the projection writer lock. A stale snapshot
cannot recreate an erased root after a newer worker has completed cleanup.
Cleanup preserves unrelated pending roots and treats previously applied erasure
tombstones as unchanged evidence, so later titles and overlays survive repeated
callbacks. Optional projection refresh failures do not reject committed native
observations or execution acknowledgments.

## API and faculty

The public API exposes `GET /v1/host/work`, `GET /v1/host/work/{id}` and
`GET /v1/host/work/summary?since=` to the owner instance credential. Other
credentials cannot inspect these records. List uses `limit` 1 through 100,
`offset` 0 through 10000, an optional root status filter, and timezone-aware
`since` and `until` bounds. The default window is seven days; the maximum is
366 days. Summary uses the same time bounds and reports unknown effect kinds.
Responses have no raw evidence text. Unknown roots return 404.

`work_records` defaults on in configuration, runtime faculties, reports and
adapter cached enablement. Off means no new work capture or projection and no
work metric section. The generic faculty report still identifies its off state.
Disabled API lists and summaries are empty with `enabled: false`.

Capture, verification and API work with mind off, lessons off and no model.
Non-Hermes hosts may submit the same authenticated evidence contract; no host
name grants trust beyond the credential. A host with no work support contributes
no work record. The faculty report shows roots per day, root status rates,
owner-corrected rate, and unknown effect kinds.

## Offline acceptance

Tests cover exact-resource mismatches, every verifier, unresolved failures,
owner and automated origins, action classification, metadata hook latency,
sealing before hashing and absence of raw arguments and results. Transport
tests cover both versions, persisted legacy retry digests, duplicates, conflicts,
downgrade, and a crash after source admission. Lifecycle tests cover overlap,
interim claims, rotation, compression, interruption, restart and late children.
Erasure tests cover whole and partial sources, derived dependencies, stale
writes, replay and rebuilding. API tests cover authority, pagination and bounds.
Faculty tests cover default, off, model-free operation and reports.

Clone and live acceptance remain outside this offline unit. No version string changes are needed for this contract.

Work admission is optional within ordinary source admission. If erasure removes
the owner request or a work fact is rejected, retained source messages still
commit and the turn acknowledges `work_recorded: false` with a reason. Projection
invalidation failures are logged after erasure; learner erasure notifications
still run.

Repeated native tool-call IDs retain every operation using a per-turn sequence in
the operation identity, and mark the envelope truncated because observation joins
cannot disambiguate reused native IDs.

Ledger work admission and erasure mark only affected root IDs in a durable dirty
queue. Each root has a generation; a worker may clear pending work only when its
rebuilt generation is still current.

Adapter action metadata uses only known tool vocabularies; unknown tools retain an
empty action. Terminal grep, rg and test probes without shell expansion, redirects
or chaining are reads (rg preprocessor options remain effectful). Hermes cron
`update` is an edit. Its list formatter exposes prompt previews and display-only
schedule/repeat fields, which cannot verify or contradict requested values.

Turn acknowledgments persist whether work was admitted. Late events for an
explicitly unadmitted root are dropped even after adapter restart. Terminal
ineligible, erased and expired-parent replies settle the event; unknown roots
may retry. Work-event retries stop after 20 attempts or 24 hours, discard their
payload, and never stop delivery of following rows.

All hosts submit job evidence as boolean field-presence/comparison facts; raw
requested or saved field values are rejected. Host outcomes use a closed enum
(`done`, `completed`, `failed`, `cancelled`, `interrupted`, `unknown`, `pending`).
Actions are limited to lowercase tokens of at most 32 characters, and unknown
tool actions are discarded before persistence.

Hermes `on_native_turn_settled` freezes retained facts when no final-response hook
did, including failed/interrupted root and child turns; repeated settlement is
idempotent. Work uses its per-turn snapshot for provenance, while answer admission
continues using the same session evidence regardless of the work faculty toggle.

The settlement fallback stores no assistant response when Hermes produced none;
its proof remains the tool evidence captured before interruption.

The settlement fallback applies only to work root candidates and bound children;
it does not add ordinary source capture for automated noncandidate turns.

Committed hands transitions queue affected task IDs durably in `hands.db`.
After commit, a worker marks only roots with the same frozen owner `origin_ref`,
using an indexed lookup. Failed refreshes retain pending tasks for restart or
maintenance retry; uncommitted hands changes never alter work. Missing-parent expiry uses
the server's first-missing-event time, independent of root age or host occurrence
timestamps. Successful parent resolution removes the event's pending wait row.

The projection stores a separate frozen-envelope revision so an old erasure
marker cannot invalidate derivatives created after that erasure was reconciled.
External execution/source refresh failures are optional and preserve their
already committed acknowledgments, including failure to look up a parent.


## Nightly titles and verdicts

The plain consolidation order is narrative, lessons, work, wishes, contradictions,
contact digests, episodes. Stages least recently entered run first; ties use that
plain order (see [nightly consolidation](CONSOLIDATION.md#what-runs-in-order)).
The `work` stage belongs to `work_records`; it can run
with lessons and the other consolidation stages off. Mind off prevents the
nightly stage. Capture, verification and reads still work without a usable model;
new rows remain untitled (`title: null`). Lessons off means no new verdict scan,
while titles and previously queued vetted verdicts can still be processed.

Titles use the shared `_call` budget and deadline machinery with the
`mind_work_titles` task explicitly assigned to the extraction role. Each night
selects at most 16 current untitled roots, independently of the nightly date
cursor, and calls the model separately for each root within the shared budget.
Each call receives only that root's owner request, at most 600 characters;
opaque labels hide transport IDs. Tool labels and other roots are excluded.
Output is capped at 768 tokens per call.
`schema/work-titles.json` and runtime validation permit only a list of titles,
each 3-8 whitespace-separated words and at most 128 characters. Unknown or
duplicate IDs, extra fields and invalid titles reject the batch. No model output
changes execution status or owner overlays. Generated secrets are sealed and the
stored title revalidated inside the source/revision/generation write fence.
Reads recheck the stored source versions and attribution. Old titles generated
from shared batches are hidden even before migration, then cleared and retried
because their original input dependencies were not fully recorded.

Untitled roots are the durable pending selection. Before the call,
`work_title_attempts` records the selected revision, attempt count and retry time.
Failures and cancellations leave titles empty and retry after 1, 2, 4, 8, 16,
then 24 hours (capped). A changed revision is eligible immediately; roots beyond
the batch limit remain pending across restart. Successful writes clear attempts.
A failed night does not replace titles with guessed completion prose.

The lesson packet retains the judged reply's source ID, complete source revision
and message hash, plus the verdict's identity. Only a single immediately
preceding assistant reply is eligible. A paired source must identify its adjacent
user input; separately retained replies must carry one exact `_supplied_inputs`
reference backed by the current `_supplied_sources` revision. Dependency references
alone, tool/system messages, multiple replies, conflicting verdicts, intervening
messages and tied source times do not establish a link. The direct input must
match the admitted root's owner-message hash and session.

After lesson vetting and before advancing `lessons.scanned`, the lesson stage
commits a deterministic verdict inbox record in the canonical source ledger with
source dependencies, root generation and the captured semantic revision. Missing
or stale candidate projections are repaired before capture; a concurrent refresh
rolls back capture so Lessons can retry without advancing its cursor. Pending
decisions wait for repair and survive refreshes that preserve their revision.
Legacy pending records without a captured revision retain the generation check.
Linked decisions retain their applied
semantic revision, so rebuilding a lost work projection restores current overlays
even after a harmless generation refresh. Acceptance commits to the canonical
ledger before projection application; an interrupted projection acknowledgment
therefore remains replayable. Inbox failure prevents progress marking. The work stage processes at
most 48 pending verdicts without a title-model call; retries have the same capped
watermarks as titles. Chronology and source versions are rechecked at capture,
application and read time, including delayed backdated messages. Newer verdicts
cannot be replaced by older retries. Erasure, attribution correction or changed
root evidence prevents stale application. `confirmed` and `corrected` remain
separate from execution status, including an unresolved execution failure.

Verdict coverage is explicitly limited to the lesson packet: six sessions and
eight retained source turns per session. The summary and nightly audit report the
limits, sessions reviewed/omitted, sessions with omitted turns, vetted/queued/
unlinked verdict counts and the review time. The existing newest-first lesson
cursor can omit older overflow sessions; these counts do not assert complete
owner-verdict coverage. Before a successful review, `reviewed_at` is null.

`GET /v1/host/work/summary` supplies the digest with `counts` (execution status),
`overlays` (`confirmed`, `corrected`, `none`), `untitled`, `titles.unproven` and
`titles.failed` (up to five each, newest first), and `verdict_coverage`. Titles and
overlays with stale source lineage are excluded. Bounds select root admission
time, including the root's current later effects and overlay. The generic mind
digest caller reads the same summary and passes one count line through the
outbox to the existing renderer. Empty work or faculty off adds no line.

The night's wish stage, which follows work in the plain order, reads the owner's work of
the last 14 days through `WorkStore.failures` on a read-only connection: each
current root's status, overlay, the tools of its acting calls in call order and
each failed call's tool and host failure kind (`failure_kind` from the admitted
envelope), leaving out a failure a later verified call to the same resource made
good. Codes only: no title, argument or result. Failed work and `corrected`
overlays become failure classes the mind may wish for a capability for; guard
blocks, unproven and partial work do not (see [MIND.md](MIND.md#wishes)).

Regression tests cover title correction and erasure, stale projections, delayed
repair, restart, changed evidence, and concurrent capture refresh.
