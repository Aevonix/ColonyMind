# Durable autonomy scheduler

Phase C slice 2a hardens ColonyMind's existing autonomy scheduler. It does not add
an evidence reducer, action authority, approval behavior, queue execution,
voice behavior, or a new external integration. The public
`register(name, callback, interval_seconds, metadata=None)` call remains
unchanged.

## Persistence and migration

Opening an existing `schedules.db` performs an additive SQLite migration. The
original `schedules` columns remain in place; failure, lease, degraded-state,
and created/updated fields are added. Two append-only tables record:

- every claimed `schedule_run_attempts` row, with its exact lease token and
  expiry;
- one terminal `schedule_run_receipts` row per attempt.

Database triggers reject updates or deletes on both audit tables. A unique index
enforces one durable schedule per stable task name. If a legacy database already
contains duplicate names, the earliest row remains canonical and every other row
is preserved under a deterministic `#legacy-duplicate-…` name, disabled with a
truthful migration reason. Existing task IDs, run timestamps, enabled state, and
cadence survive normal re-registration.

The legacy `ScheduleStore.upsert()` entry point remains available for embedded
callers, with an explicit ownership boundary. A new row may be initialized from
the supplied schedule. For an existing row, only caller-owned definition and
configuration fields (`name`, interval, callback name, enabled state, and
metadata) are updated. Scheduler-owned `last_run`, `next_run`, failure and
degraded state, and the exact lease token/expiry are preserved atomically. A
stale object retained before another process claims the task therefore cannot
make that live run claimable a second time.

## Claim and completion contract

A tick uses `BEGIN IMMEDIATE` to select and lease one due row. Claim eligibility
and update conditions both require that the schedule is enabled, due, and
unleased or expired. Separate processes therefore cannot acquire the same live
lease. Callback completion must present the exact current lease token, attempt
ID, and schedule binding before its expiry; mixing fields from two otherwise
valid claims is rejected without terminalizing either claim. Re-enabling an
already-enabled schedule is idempotent and cannot revoke its live lease. The
same lease-preservation rule applies when a compatibility update changes the
enabled flag to false before a later enable: the original attempt retains its
exact lease and is the only run that may produce the terminal receipt.

Deletion is idle-only. `ScheduleStore.delete()` returns false while a schedule
has a lease or any attempt without a terminal receipt, including inconsistent
legacy state where another writer already cleared the token. This keeps every
claimed run bound to a durable terminal outcome. An operator that intentionally
cancels a live run can disable it first; disable writes a `disabled` receipt and
clears the exact lease atomically, after which deletion is allowed. Historical
attempt and receipt rows remain append-only after the schedule definition is
removed.

Success appends a terminal receipt and atomically advances `last_run` and
`next_run`, resets failures, and clears the lease. Callback failure does not
advance `last_run`; it appends an error receipt, increments the failure counter,
and releases the task at a bounded exponential retry time. Defaults are a
five-second base and one-hour cap and can be configured when constructing the
scheduler.

If a process crashes, the lease remains visible. The first claimant after expiry
atomically appends a `lease_expired` receipt for the abandoned attempt and creates
one retry attempt with a new token. A late callback carrying the old token cannot
advance the schedule or overwrite either receipt. This is an at-least-once crash
boundary: a callback whose process disappeared after an external side effect but
before its receipt may be retried, so callbacks still need their own idempotency
key for external mutations.

The scheduler supports native async callbacks, synchronous callbacks, and sync
callables that return a genuine custom awaitable or generator-coroutine.
Awaitability classification reads exact coroutine/generator state and cached
built-in MRO/dictionary descriptors; it does not invoke generic ABC checks or
metaclass `__mro__`/`__dict__` hooks. Any metaclass-injected non-exact namespace
key is treated as opaque for the whole MRO because its equality hook can execute
during `await`; normal exact `__await__` methods remain compatible, while an
exact first-MRO `__await__ = None` continues to disable awaiting.

Callback result persistence is diagnostic: exact built-in JSON values are
canonicalized under fixed node, item, depth, text, and integer budgets. Container
subclasses and arbitrary objects are treated as opaque, so result projection
never invokes user iteration, indexing, length, `str()`, `repr()`, or metaclass
attribute hooks. Non-JSON, non-finite, or oversized values receive a bounded
projection with type, stable bounded-shape digest, reason, and useful summary.
Sets are summarized from sorted structural tokens, making their projections
stable across processes and hash seeds; opaque object identity and memory
addresses are never included. Summary whitespace is preserved exactly.
Projection completes before the SQLite writer transaction. After commit,
`tick()` returns that same persisted bounded projection, never the original
unbounded or hostile callback object.

Callback exception diagnostics follow the same rule. A type label is read by
calling the cached `type.__dict__["__name__"]` descriptor directly, and only the
base exception's exact argument tuple is summarized. This bypasses both custom
metaclass `__getattribute__` methods and metaclass data descriptors. User
exception `str()`, `repr()`, and metaclass hooks are not called. Type/message
rendering completes before `BEGIN IMMEDIATE`, including when a hostile exception
is returned to the retry path. `KeyboardInterrupt` and `SystemExit` remain
outside the scheduler's `Exception` catch and retain normal process-control
semantics.

## Degraded schedules and health

A durable schedule whose callback is absent after restart is never deleted. Its
claim receives a terminal `degraded` receipt and the row is disabled with
`callback_unregistered:<name>`. Registering that callback again recovers this
degraded state. A valid registration also repairs a row disabled specifically
for invalid legacy metadata, interval, or callback name; an operator-disabled
schedule and a duplicate-name migration conflict remain disabled.

`AutonomyScheduler.health` returns a bounded snapshot with schedule, due,
active/expired lease, attempt, receipt, and degraded counts plus the last tick
status. Expired leases or degraded schedules make the snapshot unhealthy.
`list_run_attempts()` and `list_run_receipts()` provide bounded diagnostic reads.
Lease tokens are deliberately omitted from `TaskSchedule.to_dict()` and the
operator schedule listing.

Schedule metadata is a canonical flat JSON object: at most 32 safe keys, scalar
or bounded scalar-array values, 500 characters per string, and 8192 encoded
bytes. Nested structures, non-finite numbers, oversized integers, and unsafe
keys fail registration before the callback registry or database changes.
Callback results that cannot be stored directly are represented by deterministic
digest/type/reason/summary projection JSON. Its final UTF-8 serialization is
strictly bounded to 8192 bytes, including ASCII escaping of BMP and astral
Unicode characters. The projection digest is explicitly a bounded diagnostic
shape fingerprint (`bounded-shape-v1`), not an attestation of every byte in an
arbitrarily large callback object.

