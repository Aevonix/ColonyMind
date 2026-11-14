# Session archive deployment

The sidecar owns the archive overlay and the HTTP API. `server.py` injects
`ArchiveQueueClient`; `host.py` only accesses the injected engine. The sidecar
never imports cron or adds Hermes to its Python path. With no injection, no
store, or no authenticated worker poll during the last 60 seconds, it does not
advertise `session_archive`, and archive routes return 409
`unsupported_capability` without writes.

The gateway loads our `colonymind_hermes` adapter and starts one supervised pull
task with `PluginContext.spawn_task` and `on_unload`. A private process lock
excludes other workers using the same journal. Forked children relinquish the
inherited lock and cannot poll. Unload and gateway exit cancel the task. Stock
Hermes loads plugins on a deadline thread without an event loop, so the adapter
hosts its supervised task on a daemon thread's asyncio loop when needed.
Each engine unit uses a fresh subprocess of the gateway's own interpreter, with
a five-second deadline including imports, recovery and receipt persistence. A
daemon supervisor holds the gateway's cron RLock and checked flock throughout.
The child borrows the passed flock and uses core's nested-lock protocol. On any
timeout or supervisor error the child is killed and reaped before exclusion is
released. Cancellation releases the async task promptly while the supervisor
retains the worker lease until the child has stopped. Ordinary scheduler claims
can then drain during Python's executor shutdown. An interrupted unit leaves its
durable intent for next-start recovery; no deadline-generated receipt is sent.
The child uses the same source paths, effective cron-store paths, settings and
profile environment, without forking the threaded gateway's Python state.
It arms its own termination timer against the same monotonic deadline and
inherits the worker lease, so gateway exit cannot leave an unbounded mutator or
admit a reload while the child can still write.
Startup requires an
accepted, active unload registration. Kanban workers do not start it.

The task uses the adapter's existing authenticated HTTP direction:

- `GET /v1/host/sessions/archive/plans?wait=25` records one heartbeat per incoming
  poll and leases one owner-scoped plan for 60 seconds. Empty queue checks are
  read-only; a disconnected poll stops before claiming again. Capability expires
  exactly 60 seconds after the last incoming poll.
- `POST /v1/host/sessions/archive/plans/{operation_id}/receipt` submits its result
  with the opaque claim returned by the poll. Operation IDs are URL-encoded.

Both routes require authenticated owner authority and modern `manage`, or the
dedicated legacy scope `sessions:archive-worker` (or `*`). Read and write session
scopes alone cannot claim plans. A receipt is bound to the credential that
claimed the plan. An identical duplicate returns 200; a different result returns
409 `idempotency_conflict`.

The plan carries the original owner and `sessions:write` decisions. Before any
cron mutation, the adapter checks those decisions against its configured
`owner.contact_id`, reads Hermes `state.db` **in mode=ro**, and checks both
session/origin bindings in `gateway_routing` and their sender ownership. Missing,
ambiguous or mismatched bindings fail closed; ownership refusals name the
session. State lookup failure is `source_unavailable`.

The adapter packages the existing `session_archive/reparent.py` and `store.py`
sources as `colonymind_hermes.archive_core`, using relative imports. It imports no
sidecar package to run them. The complete engine operation runs in the gateway's
venv, against the same cron store and delivery configuration as the scheduler:
resolver, planning, public `cron.jobs.update_job`, readback, final proof and
compensation all execute under Hermes' reentrant cron-store lock and the journal
write lock, in that order. The adapter establishes the shared flock itself and
returns `busy` if it cannot acquire exclusivity, rather than using core's
fail-open timeout behavior. Ordinary core writers participate in this section.
Hermes core is unchanged; writers that bypass its locks remain outside it.
The adapter's source directory has a relative `archive_core` link to these same
sources, so stock directory loading also works without an installed-package
finder or a second copy of the engine.

The adapter journal is next to the turn outbox, with `-archive.sqlite3` appended
to its stem. With default settings this is
`$HERMES_HOME/state/colonymind-turn-outbox-archive.sqlite3`. It retains durable
intents, attempted writes, operation IDs and receipts. **Keep this file across
upgrades and restarts.** A reclaimed plan returns its stored result; a crash
during a mutation recovers before fresh authorization, using durably recorded
authority and verified current cron origins. Missing bindings and transient
sender lookup failures cannot hide an earlier write. Unresolved attempted jobs
are persisted as `uncertain`, with `half_applied` and `receipt_incomplete`.
A new archive operation can retry recovery after the blocking condition clears.

The sidecar persists the request by `operation_id` in its existing
`colonymind-session-archive.db`, waits asynchronously up to 15 seconds, and commits
the overlay atomically with receipt acceptance. A timeout returns 503
`source_unavailable` with `pending: true` and `operation_id`. The operation keeps
running; retry the identical request with the same ID to obtain its receipt.
Receipt reads expose pending operations and reported half-applied jobs. The
adapter journal remains the recovery authority for cron writes; receipt reads
do not revalidate current cron delivery. Old overlay receipts replay unchanged.

## Installation

Install the sidecar wheel into the sidecar's environment, here its pipx venv:

```sh
~/.local/pipx/venvs/colonymind/bin/python -m pip install --force-reinstall --no-deps /path/to/colonymind-wheel.whl
```

Install the `colonymind_hermes` wheel using **the interpreter that launches the
Hermes gateway**, then restart both services:

```sh
/path/to/gateway/venv/bin/python -m pip install --force-reinstall --no-deps /path/to/colonymind_hermes-wheel.whl
```

No port, listener, credential or credential rotation is needed. Keep the existing
`plugins.colonymind.sidecar_url` and `plugins.colonymind.key_file`; the latter
defaults to `~/.colonymind/api.key`. `COLONYMIND_API_KEY`, when already supplied,
continues to take precedence. For a carried scoped credential, add
`sessions:archive-worker` to its existing grant configuration without changing
its secret. The instance key and modern owner credentials with `manage` need
no new grant. The adapter's `owner.contact_id` must match the sidecar's owner
identity and its owner handles must resolve both sessions' bound senders.
`--force-reinstall` replaces a wheel of the same version that is already
installed; `--no-deps` leaves the installed dependencies as they are.

An old adapter sends no heartbeat, so a new sidecar keeps archive unadvertised.
A new adapter polling an old sidecar receives 404 and backs off quietly for
30 seconds. The capability can remain advertised for at most 60 seconds after
the last successful worker poll.

## Verification

The session archive requires Hermes 0.21.7 or newer. Its complete stock install
uses Python 3.14, with the adapter in that environment and the sidecar in a
separate Python 3.11 to 3.13 environment. The adapter suite is qualified on
Hermes 0.21.7 and also runs on Hermes 0.23.1 in the hosted checks; see
[compatibility](COMPATIBILITY.md) for the hosted checks'
dependency-marker adjustment on older interpreters and the measured installation
results. `init`, `upgrade` and `doctor` require `>=0.21.3,<0.24` in package metadata
or, when metadata is `0.0.0`, the owning Hermes checkout's `install-stamp.json`
`baseVersion`. Hermes 0.24 is outside that range. Adapter qualification does not
establish an end-to-end installation or archive qualification for another release.

Use `COLONYMIND_TEST_SIDECAR_PYTHON` to point the adapter integration test at a
separate sidecar-only venv. The test asserts that interpreter cannot import
`cron`, starts the sidecar's actual authenticated routes, runs an adapter worker
in the Hermes-capable interpreter, checks capability activation, archives a real
cron job, and verifies receipt replay and inventory. Other tests exercise real
core compensation after refusal and after interruption, owner and scope refusal,
heartbeat expiry, async waiting, duplicate receipts and worker unload/backoff.
Set `COLONYMIND_TEST_ARCHIVE_WORKER_PYTHON` to a separate Python 3.14 Hermes
environment with the adapter wheel installed. The archive suite's second
interpreter, selected by `COLONYMIND_TEST_HERMES_PYTHON`, must also be Python 3.14.

For the full sidecar gate, also set `COLONYMIND_TEST_HERMES_PYTHON` to a disposable
private Hermes interpreter and `COLONYMIND_TEST_ADAPTER_SOURCE` to the built
adapter wheel. Excluding `test_init.py` does not exclude its helpers: other init
test modules import them and perform real package installations into that child
interpreter. Keep that child separate from the archive integration worker's venv.
