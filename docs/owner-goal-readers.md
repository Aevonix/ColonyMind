# Owner-goal read boundary

`colonymind.owner_goals.reads` is an internal, read-only boundary for an explicitly
configured Hermes profile. It adds no routes, scheduler, mutation, activity store,
migration manifest, ledger or notice. All callers must already have authenticated
owner and audience context. No default home or contact-based ownership fallback
exists.

## Job catalog

`HermesOwnerReader.catalog_snapshot(owner_id)` reads one native jobs document and
one trusted destination-registry snapshot. `list_owner_cron_jobs` pages that
ephemeral snapshot in profile/job order. These snapshots are not durable migration
manifests. Consumers must retain and inspect their server-side exclusions.

The injected `DestinationRegistry` is an authority boundary, not a JSON request
or a scheduler destination parser. Its records bind the exact profile, job,
routing digest and evidence version. Each destination must independently prove
the platform, account, chat, optional thread, directness and sole recipient. The
registry must resolve all effective routing lanes, including defaults and failure
destinations, and prove there are no additional recipients. Job names, prompts,
profile membership and delivery strings supply no ownership proof.

Known foreign and disabled destinations are excluded before pagination. Missing,
ambiguous, shared, stale or unverified evidence excludes the job and records an
unresolved reason. Any unresolved eligibility makes enumeration return typed
503 `owner_goals_unavailable`, carrying exclusions internally. It cannot return
a successful exhausted page that could authorize migration completion. The
registry protocol has no permissive production implementation: existing contact
aliases lack the account-scoped routing authority required to implement it.

Handles retain object and evidence versions and native enabled/recurring facts.
Paused jobs can qualify; one-shot schedules and repeat-once jobs are not recurring.
Run/activity access revalidates the current handle and owner audience, so stale
ownership, evidence or job configuration cannot authorize reads. Evidence and
internal diagnostics must never be projected as public artifact fields.

## Native run-history limitation

The native `cron.executions` ledger retains execution IDs, times and delivery outcomes,
but no per-execution job/destination authority snapshot or exact output reference.
Terminal ledger records are pruned. `cron.delivery_queue` associates a job snapshot
and content with an execution while pending, then erases both at terminalization
and bounds retained outcomes. `cron.jobs.save_job_output` writes timestamp-named
files with independent retention; these names cannot prove execution identity or
historical ownership. `cron.owned_output.snapshot` is bounded and explicitly
requires callers to attest ownership; it also opens native write transactions.
It cannot establish complete authorized run history.

Consequently `list_owner_job_runs` validates pagination, current handle and
audience, then returns typed 503 `owner_goals_unavailable`. It does so even for an
empty ledger or an offset beyond retained records. A terminal outcome, current
destination or large output file cannot repair missing historical evidence.
Successful `OwnerRun` pages and original-output access remain unimplemented
pending a native retained-history boundary with exact execution/output binding,
versioned per-run account/destination/owner evidence, recorded job name and
delivery outcome, audience enforcement, and explicit gap/retention coverage.
No success-path run qualification is claimed.

## Activity

`merge_activity` consumes authorized, ordered source pages, starting each at zero.
It compares absolute times, deduplicates stable source identities, applies the
requested offset only to the merged stream, and reads enough for continuation.
It buffers one bounded page per source and the result, plus visited identities.
It rejects broken pagination, source failures and oversized single entries;
multi-entry pages shrink with integer continuation at the response byte budget.
Source readers must validate audience/tier and attest their required history
before supplying Activity entries. No content is persisted or generated.

`list_owner_activity` uses only the current attached handles and the supplied
authenticated progress reader. With the reviewed native source, an attached cron
source makes activity unavailable; detaching it removes that source from the
derived view. Progress-only pages work. Successful cron-run activity is subject
to the same native dependency as original run output.

## Qualification

`sidecar/tests/test_owner_goal_reads.py` checks the native pin and launches a fresh
interpreter with a temporary `HERMES_HOME`. Native `cron.jobs.create_job`,
`cron.executions` transitions, `cron.delivery_queue` transitions and
`cron.jobs.save_job_output` write the actual profile, including a two-mebibyte
output. There are no hand-written scheduler formats, model calls or network
deliveries. Registry fixtures are independent server authority records.

Tests verify destination-proof exclusions, non-owner/ambiguous/unverified/account
evidence, read immutability, stale handles, audience checks, safe pagination and
missing-evidence 503 responses. Separate progress-source tests exercise ordered
merge, ties, deduplication, offsets, byte limits and source failures; they are not
claimed as native run-output qualification. Full-suite counts and baseline node
differences are recorded in the change report outside this source tree.
