# Hands

Hands are durable attempts at a task, with explicit intent versions, evidence,
effects and obligations. The shared types, wire models and database schema
support the controller and plugin runtime described below.


## Shared contracts

The sidecar defines the task and hand records in `colonymind.hands.types`, API
models in `colonymind.hands.wire`, and an idempotent schema migration in
`colonymind.hands.store`. The wave 1 API handlers run the controller, evidence,
approval and obligation operations. No reserved Hands operation returns 501.
The plugin's `register(ctx)` entry points install the dispatch, guard, middleware and front hooks described below.
The plugin also exports a resolver callback registration seam for host roles;
its signature and source-loading boundary are in the
[plugin README](../plugins/colonymind-hands/README.md).

Configuration lives in `hands` and the faculty flag `mind.faculties.hands` in
`colonymind.yaml`. The faculty defaults to on; `hands.max_running` defaults to 1.
The flag takes effect without a restart: the plugin reads the file on each
dispatch, and the controller re-reads the flag on each plan and sweep when the
file has changed (an unreadable file keeps the last value read).
The faculty report has a Hands section that reads the controller's retained
records. Its paired benchmark arm is `all-on-hands`, a no-op contrast
while the benchmark runs no hands executor.
`hands.platforms` is an empty allow-list, so no platform is admitted by default.
Hosts supply platform support, route contracts, role mappings and family limits.
Detached completion delivery must be verified before a platform is admitted.

Numeric defaults are configured enforcement bounds, not measured guarantees:

| Setting | Assumption |
|---|---|
| `budgets.attempt` | Per-kind wall time, tool calls, searches and tokens; effort 3 doubles wall time and tokens. Tokens are uncached prompt plus output (see Budget accounting) |
| `budgets.final_report_*` | After a token budget stop, one final report call of at most 8,192 output tokens (`final_report_tokens`) within 300 seconds (`final_report_seconds`) |
| `budgets.task` | Three times the first attempt's token allowance, 3 hours, 6 hands |
| `budgets.day` | 40 autonomous hands; 3,000,000 tokens per family unless the host overrides it |
| `queue` | 6 queued hands: at most 4 owner and 2 own; deadlines 900, 1,800 and 3,600 seconds by priority |
| `health` | Poll every 5 seconds; no engine waiters and KV use below 0.8; one hand per family by default |
| `lease_seconds` | Reservation lease 90 seconds |
| `dispatch_start_seconds` | Look up a dispatched hand with no start after 120 seconds |
| `open_tasks` | At most 6 own-origin and 4 mind-origin tasks; owner tasks have no count limit |
| `brief` and `digest` | Brief limit 6,000 tokens and digest 900 characters; recall 2,500 tokens and identity 900 characters are fixed assembly bounds |
| `checks` and `cascade` | Check timeout 300 seconds; cascade limits 1,000,000 per task and per rolling hour |
| `retention` | Events and settled task rows 36,500 days, with the safety exceptions below |
| `obligations` | Outstanding answers become overdue after 3,153,600,000 seconds and appear as overdue in the digest |
| `locks` | At most 2 waiters per resource |
| `spool` | Replay and progress coalescing every 10 seconds |
| `cooldowns` | Two consecutive failures pause a task for an hour; a third parks it; repeated resource failures pause for 6 hours |
| `notices` | 30 routine and 20 actionable messages a day, merged over 60 seconds; at most 200 characters per line |

Upkeep shares the autonomous queue deadline, an implementation assumption.
Daily token limits and family caps use generic defaults with host-provided
overrides. No model family, endpoint, owner identity or delivery destination is
built into this seam. Thinking stays in the background until request-budget and
prefix measurements support an inline option. The direct line defaults to off.

The cascade, retention and overdue defaults deliberately allow ordinary use until
the host supplies measured values. They are staging assumptions: one million
continuations per task/hour and approximately a century of retained history and
unmarked debt. Hosts should tighten them to their observed workload before release.

Every admitted continuation counts as a cascade, including a retry, verification
or a change of kind/route. Initial attempts do not count. Admission at the cap is
allowed; the next continuation is refused with `cascade_task_limit` or
`cascade_hour_limit`. Refusals persist as hands and events, appear in `/hands`,
and use the existing actionable-notice channel. Queued continuations count once,
when planned, and are not charged again when reserved. A queued continuation is
admitted for the per-task limit even if it is refused later; that later refusal
excludes it from the rolling-hour count. Plans refused immediately do not
consume the limit. Counts survive replay, restart and task revision; the rolling
hour is shared by all tasks and controller connections. Zero refuses every
continuation. These limits supplement the existing attempt/task/day budgets.

The sweep marks an unpaid obligation overdue only after its current intent's age
exceeds the configured seconds. A revision starts a new obligation; a refusal or
repeated result does not reset its age. The sweep records one overdue event and
queues an actionable notice. Digest reads also label overdue debt immediately.
An overdue obligation remains open until normal communication or cancellation;
overdue status does not accept a result or execute another hand.

Retention keeps records at the exact configured age and expires older records
in bounded batches at most hourly. The last run, including an empty run, is
recorded in `maintenance`. Overdue and retention each use a savepoint so their
failures are logged without rolling back expiry or admission. Event expiry
applies to settled tasks; row expiry
removes the whole settled task and its attempts/dependent operational rows.
Open debt, unfinished attempts, unresolved effects, open questions, pending
permits or notices, provenance derivations and dependency targets retain their
supporting history. Recent cascade, daily-budget and resource-cooldown accounting
also retains its rows while those counters can affect admission. Retention
never cancels work or closes an obligation. Retention days must be at least 1.
Deleting an owner-turn task preserves its origin reference for a committed Work
root refresh, including a retry if that refresh fails.

Expired replay answers become HTTP 410 tombstones; their keys cannot recreate
work or replay a lease. Minimal child-session identities remain tombstones so
expiry never promotes a child to owner authority. These tombstones and provenance
ancestry outlive operational retention. `GET /v1/hands/retention` exposes the last
committed expiry counts, cumulative counts and the limits used; one summary row
is retained. The sweep response contains its own expiry counts.

The following is the reader inventory for the settings in the
cascade, retention, obligations, brief, health, stalls, contacts and vector-store
groups. Retired front, review and triage defaults are dropped with upgrade notices.
Readers below affect product behavior; declaration and validation are not readers.

| Retained setting | Product reader |
|---|---|
| `hands.cascade.max_per_task` | `colonymind.hands.cascade:reason` |
| `hands.cascade.max_per_hour` | `colonymind.hands.cascade:reason` |
| `hands.retention.events_days` | `colonymind.hands.retention:RetentionOperations._prune_retained` |
| `hands.retention.rows_days` | `colonymind.hands.retention:RetentionOperations._prune_retained` |
| `hands.obligations.overdue_seconds` | `colonymind.hands.obligations:ObligationOperations.obligation_overdue` |
| `hands.brief.max_tokens` | `colonymind.hands.controller:Controller._brief` |
| `hands.health.poll_seconds` | `colonymind.hands.health:EngineHealth.refresh` |
| `hands.health.max_waiting` | `colonymind.hands.health:EngineHealth.admit` |
| `hands.health.max_kv_fraction` | `colonymind.hands.health:EngineHealth.admit` |
| `hands.health.default_family_cap` | `colonymind.hands.admission:reserve` |
| `hands.health.family_caps` | `colonymind.hands.admission:reserve` |
| `hands.health.running_limits` | `colonymind.hands.health:EngineHealth.admit` |
| `hands.health.unreadable_owner_cap` | `colonymind.hands.health:EngineHealth.admit` |
| `hands.stalls.generation_multiplier` | `plugins/colonymind-hands/budgets.py:_clock` |
| `hands.stalls.tool_grace_seconds` | `plugins/colonymind-hands/budgets.py:_clock` |
| `ContactsConfig.sqlite_path` | `colonymind.contacts.store:SQLiteContactStore.connect` |
| `ContactsConfig.group_promote_min_interactions` | `colonymind.api.routers.host:scope_promotion_candidates` |
| `VectorStoreConfig.data_dir` | `colonymind.plasticity.recall_qualify:run_command` |

Upgrade note: `colonymind init` used to copy all defaults into `colonymind.yaml`.
`colonymind upgrade` first keeps a timestamped backup under `backups/`, then
removes these seven retired YAML keys and prints each removal:

- `hands.brief.recall_tokens`
- `hands.brief.identity_chars`
- `hands.health.shedding_window_seconds`
- `hands.health.front_ttft_p95_seconds`
- `hands.health.voice_decode_p50_tokens_per_second`
- `hands.health.recovery_seconds`
- `hands.stalls.approval_lapse_seconds`

Loading an old file logs one upgrade warning per retired key and ignores it;
configuration writes still reject retired keys with the same upgrade message.
The same upgrade also removes `hands.cascade.max_per_task: 2`,
`hands.cascade.max_per_hour: 6`, `hands.retention.events_days: 30`,
`hands.retention.rows_days: 180` and `hands.obligations.overdue_seconds: 86400`
only when the stored integer equals that exact old init default. Staging defaults
then apply. Every other value is kept as the operator's choice. Each removal is
printed and the same backup preserves the original file.

Contact constructor arguments for enrichment, import, vCard, retention,
default-tier, automatic-merge and brief flags were also inert and are removed;
they are rejected by `ContactsConfig`, rather than treated as YAML settings.
Contacts retain their database path and group promotion
interaction threshold. The retired embedding cache-size, batch-window and
quantization variables are described in [Embedding generations](EMBEDDING-GENERATIONS.md).


## Controller

The controller persists tasks, attempts and replay answers in the shared SQLite
schema. Every mutation takes an `Idempotency-Key` and a `BEGIN IMMEDIATE`
transaction. Independent database connections therefore share admission counts.
A key repeats its first answer; using it for a different operation is refused.
Executor writes require the recorded epoch and an allowed prior state. Planning
has no epoch field in the frozen wire contract: the reservation starts unbound,
and its secret lease token authorizes the first dispatch to bind the epoch.

Intent versions retain goal hashes and criteria, without storing brief prose.
Revisions replace neither budgets nor earlier criteria. Old results become
`superseded`; explicit acceptance for the current version records a judgment and
keeps the evidence level. `Controller` installs `evidence.evaluate_result` and
uses `acceptance_outcome` after its intent guard. Planning builds the brief from
the saved acceptance and configured attempt budget; a task whose required text
exceeds the brief budget is refused with `brief_too_large`. No effect is executed by
the controller.

Attempt allowances, cumulative task tokens and wall time, the six-hand task
limit, initiative daily counts and family token counts use the configured
assumptions above. Reservation admission consumes an attempt even if dispatch
later fails. Plugin usage events are monotonic cumulative counters, with the
actual family required for token increments; the controller attributes those
increments to the local day. Owner work is exempt only from initiative daily
limits. Own and mind origins cannot claim owner priority. Two consecutive
failures pause a task; the third parks it. Failures on the same resource across
three tasks pause that resource. The plugin enforces tools and requests locally
and acts on controller stop events. Full shell access remains best effort.

`Controller.events` accepts activity, checking and usage events. Plugin event
sequences map to even stored sequence numbers; controller events use odd ones.
Payloads pass through secret redaction and the schema's size bounds. Effects
require an intent before an outcome. Reconciliation checks executor identity,
marks abandoned work `lost` or `unknown_effect`, and never retries it. An
unresolved effect blocks continuations through either a hand or a task ID until
an explicit reconciled continuation is recorded.

`Controller.erase_parent(session, key)` clears branch handles, questions,
answers and relay derivations, retracts other derivations, and tombstones replay
answers. Erased keys return 410 and cannot recreate work. It calls `Provenance.erase_parent` before clearing controller ancestry,
retracting the dependent source entries and lessons.

`colonymind hands list`, `show`, `log`, `tasks`, `obligations` and `report` read the
local database without migrations or writes. `stop` and `revise` use the
owner-authenticated sidecar API. Reports show retained states, evidence levels,
unconfirmed effects and pending notices; they do not claim measured uplift.
The existing owner-only router rejects anonymous and non-instance credentials.

API inspection returns all matching rows; names use a
kind plus an ID suffix when omitted; hosts supply `hands.reserved_names` for
identity, contact and tool names. Structural defaults also reserve common
commands. Tests use no live executor or external message sink.


## Admission

Reservations count hands, including checking and integration, separately from
Hermes executor slots. Every start holds a family slot and any declared write
resources. A lease expires after the configured 90-second assumption. The
maintenance loop runs at the health poll interval and promotes queued work
through the same checks. Queued work uses state `planned` and activity `queued`.
The four-owner and two-own queue assumptions remain bounded; mind work waits in
its external executor instead. Expired queue deadlines stop waiting attempts.

Admission allows one resource waiter by default;
`hands.locks.admission_waiters` may opt into two, never more than the schema's
hard limit. All resources are checked before any is acquired. Coding without
explicit resources needs a working directory and locks its repository root or
path; ops defaults to `ops:local` until resources are specified.
A coding task with no working directory, no resources, no need beyond `shell`
and no test command is shell work, so it is planned as ops: what a task changes
is a structural fact and outranks the requested kind. The hand records
`kind_adjusted` (from, to, reason) in its decisions and as an event, and its
continuations stay ops. Any other coding or integrate plan without a working
directory or resources is refused with `coding_requires_workdir_or_resources`.

Hosts configure `hands.routes` as role-to-provider chains and supply versioned
`hands.contracts`. The static matrix fixes effort at 2. Unknown providers,
missing modalities, insufficient context or unqualified required tools drop
providers from a chain. Verify routes exclude known producer families. Context
estimation conservatively counts UTF-8 bytes as tokens; hosts can add expected
growth per kind through `hands.context_growth_tokens` (default zero) before the full brief is assembled.

`hands.health.metrics_urls` maps families to host-supplied metrics endpoints.
The reader polls every five seconds, an assumption, and checks running requests,
waiting requests and KV use across all callers. A missing metric invalidates the
reading. Counts sum across engine labels and KV uses the maximum, an assumption.
Hosts must provide measured `running_limits` and `family_caps`. Stale or
unreadable readings allow only one owner hand per family; other priorities wait.
This module does not queue the front or voice and does not open sockets without
configured endpoints. The executor runtime checks headroom and platform delivery.

After queue promotion, `GET /{hand}` exposes the lease in
`decisions.admission` because the frozen hand wire record has no reservation
field. The original plan replay remains immutable. The sweeper returns
`lookup` IDs for dispatched work without a start after the configured deadline;
the plugin supplies the executor ledger view through reconciliation. The lazy
runtime starts maintenance on first use and provides `close()` for shutdown.


## Obligations

Owner-turn plans create obligations bound to the originating message. They
remain visible in every owner session, including new sessions, until
communicated, delivered, cancelled with an owner message reference, or superseded
by revision. A revised task gets a fresh obligation.

Result ingestion records `result_ready`. `Controller.review` advances it only
when a front turn both rendered the result and ended successfully. A settled
hook or a completion-queue acknowledgment is insufficient. Automatic acceptance
is recorded separately; explicit acceptance after review advances to `accepted`.
`Controller.communicate` requires that successful review turn and the original
platform/chat, and a silent reply leaves the obligation open. The explicit
`told` route records communication in another turn. Delivery remains `unknown`
until the delivered route records a platform message ID, meaning platform
acceptance only. No transport is implemented here. Any host RCS completion
adapter must restrict delivery to the owner's own thread, the original
conversation and the agent's completed work.

The digest stays within the configured 900-character assumption, shows evidence
with scope when available, and folds overflow into a count. Structured items
retain every open obligation. `Controller.owner_message` marks open tasks from
the same origin chat by message identity, without matching message words.


## Notices

The generic emitter stores home-channel lines and immutable notice IDs. It
never sends messages. Routine and actionable budgets are independent: the owner
settings are 30 and 20 messages per local day. Lines sharing a budget and
60-second merge window share `queued_at`; the host sink sends each group as one
message and acknowledges each line with the same receipt. Polling exposes only
completed merge windows and preserves unacknowledged IDs regardless of cursor.
Daily delivery selection also accounts for older pending groups.

Duplicate IDs replay their first answer. A sent ID cannot be rebound to another
receipt. Lines over the 200-character assumption or containing newlines are
refused; host-authored `hands.notices.forbidden_patterns` may impose additional
text validators and defaults to empty. Secret redaction runs before persistence. Capped lines remain in
the event log. The strict actionable cap also covers overflow: the controller
emits no extra hourly messages after spending it. `hands.timezone` may specify an IANA zone; otherwise local
process time sets midnight. Conversational updates remain discretionary.

## Plugin

The source plugin now wraps `delegate_task` once and serializes its admission
with both native async dispatch functions, including manual cron dispatch. Each
task becomes one hand and one background call. Admission requires an owner
session, an explicitly allowed platform, a declared push completion consumer,
a valid route contract and a free slot that leaves native headroom. Reservations
are released on refusal. An unexpected native inline result is retained and
recorded as `ran_inline`; the plugin never retries it automatically.
The wrapper also shows the task fields of the wire contract (`kind`, `effort`,
`risk`, `needs`, `accept`, `touches`, `workdir`, `name`, `priority`) on
`delegate_task`'s task items, through the tool entry's per-call schema
overrides, so the front can set them; Hermes' own schema offers only the goal,
context, output schema, images and group, and a task with no kind is a deep
(read-only) hand. Closing the runtime restores the native schema.

A task from a session whose platform is not in `hands.platforms` is refused
(`platform`) before anything is planned. A plan the controller refuses returns
`controller_refused` with the controller's stable code as `detail`, the field
names of a validation refusal, and a fixed `hint` naming the correction (for
example, kind ops for shell-only work). No value from the request is echoed.
A kind the controller adjusted is returned as `kind` and `kind_adjusted`. A plan waits up to 15 seconds for the
controller (it builds the brief, with recall, before it answers; an
assumption), other controller calls two.

Queued work holds no slot. The controller's sweeper re-runs admission for it and
reserves a lease; the plugin's maintenance pass (every replay interval) reads
that reservation and starts the hand through the same executor checks, with the
parent agent and context of the turn that planned it. While Hermes still holds
its headroom slot the pass waits for the next interval within the lease; without
the parent agent the reservation is released, and nothing runs inline.

The callback registered through `register_route_resolver(fn)` accepts role and
effort and supplies a
native delegation policy or a mapping with `credentials_cfg`. Without a callback,
the resolver reads Hermes' native policy. Primary and fallback providers require
contracts under `hands.contracts`; unsupported modalities, unqualified tools or
insufficient context refuse or remove that route. Credentials stay in memory.
The request middleware uses the actual route after fallback for audio admission
and records the families and contract versions that served the work.

Attempt counters are local, including searches, repeated calls, structurally
unchanged search results, tokens and wall time. Child task ids retain the same
hand across session rotation. Queued work is not stale. Generating and executing
work have separate clocks, with one status steer before a stale stop. Per-family
request admission waits locally and remains cancellable. Research, deep and
planning hands use explicit tool allow-lists; coding, operations and integration
retain the assistant's shell. Named outbound, secret, memory mutation and ask
approval tools are blocked in hand sessions. Shell restrictions remain best
effort. Mind-origin work follows the same kind policy and budgets.

`colonymind_hands` exposes list, show, answer, steer, stop, name, revise,
told, accept and continue. `colonymind_ask` records internal questions; a blocking
question immediately prevents later tools. `/hands` uses Hermes' normal command
path between turns. `/stop` remains the running-turn interrupt. Controls check
the calling owner session before registry helpers. Local inspection, steering
and stopping remain available when the controller is unavailable.

Stops and steers the runtime decides itself use `Runtime.system_stop` and
`Runtime.system_steer`, which have no owner-session check: budget, wall, stale,
queue and accounting stops, journal-fault stops, and the wrap-up, status and
no-new-information steers. No session asked for them, and they run on delegate
and maintenance threads where the owner's session may not resolve. A system
stop keeps every other effect of a stop: the journal entry, the executor
interrupt, recorded-process cleanup and the local status. `Runtime.stop` and
`Runtime.steer` keep the owner check for every stop or steer a session asks for,
including the `colonymind_hands` tool and a parent's `/stop`. A system stop never
raises into the hook that noticed it.

Critical transitions and effect intents/outcomes use mode-600 JSON lines with
fsync before execution or delivery. FIFO replay preserves idempotency keys and
runs at startup and every ten seconds, a design assumption. After two failed
controller calls in thirty seconds, a design assumption, degraded mode admits
only one owner hand with a minimal brief. Local outage ids are adopted through
the existing plan response when the controller recovers. A refused adoption
stops the local attempt and leaves its records pending for reconciliation.
Turning the faculty off retains local controls at the launch limit and defers
new controller plans. A failed spool write refuses new work and stops writers.
Reconciliation reads the native durable ledger. An unconfirmed effect becomes
`unknown_effect`, including a crash before dispatch acknowledgment, and cannot
continue without explicit reconciliation. Process cancellation only targets
recorded handles whose owner, PID and process start identity still match.

If any plugin module fails to register, delegation is replaced by a refusal before the error reaches Hermes, so
a partly loaded plugin never leaves the wrapper running without its guards. The notice caps and timings, the
degraded-mode threshold, the queue deadlines, the stall grace, the missing-start check and the replay interval are
read from the `hands:` block (`notices`, `degraded`, `queue`, `stalls`, `dispatch_start_seconds`, `spool`).

Notices require a host-registered `register_notice_sink(fn)` callback. It receives
`delivery_id` and immutable notice rows and returns a `platform_message_id`.
The emitter merges for sixty seconds and enforces separate daily limits of
thirty routine and twenty actionable messages, with an hourly overflow summary.
These are design assumptions. Delivery ids and receipts survive retries and
midnight. No destination is built in. Conversational updates remain discretionary.
RCS must remain off the platform allow-list until the host proves completions
reach only the owner's own thread, for completed work from that conversation.

### Budget accounting

A hand's token counter charges what each model call costs: prompt tokens the
provider did not serve from its prefix cache, plus output tokens. Cached prompt
tokens come from Hermes' `post_api_request` usage summary (`cache_read_tokens`)
or the provider fields it reads (`prompt_tokens_details.cached_tokens`,
`input_tokens_details.cached_tokens`, `cache_read_input_tokens`,
`prompt_cache_hit_tokens`, `cached_tokens`). Cache writes were prefilled and stay
charged. A response without cache fields is charged in full, as before. The hand
also keeps `raw_tokens`, every prompt token on every call plus output, which is
what the counter measured before; both go to the controller in the `usage` event.

This changes what the budget numbers mean. Before, a hand whose cached context
grew was charged its whole context again on every call: 11 research calls growing
from 15.5K to 56K tokens counted about 436,000 tokens, of which about 64,000
were uncached prompt and output, and crossed the 400,000 research budget. The
configured budget values are unchanged; whether to lower them for the new
measure is a separate decision.

### Budget stops

A token budget stop ends with the hand's own report. The stop is journaled at
once and the hand's tools are refused with "write the final report now with what
you have", but the executor is not interrupted. Its next model call is its one
final report: tools are removed from the request, output is bounded by
`hands.budgets.final_report_tokens`, and it takes a provider slot like any other
call. A provider error before any reply leaves that call to Hermes' own bounded
retry; once a reply arrives, every later call is answered locally with the report,
or with why the hand ended, and never reaches a provider. If the report has not
finished within `hands.budgets.final_report_seconds`, the clock interrupts the
executor. The result is submitted with status `partial`.

Other runtime stops (wall, stale, queue, accounting and journal faults) interrupt
the executor at once; the wall budget's report prompt is its 80% wrap-up steer.
A model call that still reaches the hand after a wall, stale, queue or accounting
stop is answered locally with the reason, so Hermes ends the turn instead of
retrying an empty response and failing the hand. A hand whose executor ended on
that local reply submits `stopped` (`failed` after a stale stop).

The controller takes the result of a hand stopped for its attempt, task or day
budget, or for the plugin's `token_budget` or `wall_budget`: the hand stays
`stopped` with the result's status as its outcome, its summary and evidence are
recorded, and the task waits for the owner's decision whether to continue. A
result after any other stop is still refused.


## Host launch

Reserved for host configuration, completion routing and launch package checks.


## Front

The front watchdog reads the adapter's pending slot
and the runner's FIFO for its own session. A waiting non-internal owner message
redirects a running completion review. It neither consumes events nor wraps
Hermes internals; busy mode remains `queue`. Binding includes session, turn,
agent identity and runner generation. Hermes' context-local agent binding covers
the short interval before the runner promotes its pending sentinel.
`hands.front.watchdog: false` turns the watchdog off at its next check without
a restart; request middleware and the ingress ledger stay active.

The memory provider uses the front's
exact session and turn classification to skip recall, contact warm-up and clock
work for a review. Completion-looking owner text cannot select this path. The
answer hook skips owner-answer lookup for reviews while retaining the local
turn record needed for answer provenance. When an owner already waits at review
start, the front also skips the controller digest.

The hands `llm_request` hook checks the same session-bound pending slot and FIFO
before sending a review request. If the owner is waiting, it records a drop;
the existing `llm_execution` hook returns a local terminal `[SILENT]` response
without calling the transport. Execution checks again for arrivals between the
hooks. This uses native middleware because a request-envelope return cannot
cancel a call, and interrupting a non-streaming call alone can still start its
worker. No provider usage is reported. Session, turn, agent and generation must
still match; the watchdog rollback also disables new request drops.

A dropped review earns no rendered or communicated receipt. Its exact native
completion message is scheduled through internal gateway injection only after
the owner's gateway turn settles and no further owner event is waiting. The
retry remains pending until a matching review actually starts, so a later owner
settlement can retry a failed route, at most three times (an assumption);
after that the retry is dropped and counted as `review_retry_abandoned`, and the
obligation stays open with no review receipt. Injection acceptance is a scheduling
receipt, not a delivery receipt. The retry rechecks native
completion provenance and coalesces eligible results as usual. Busy mode stays
`queue`, and neither native queues nor Hermes internals are modified.
If another owner turn makes the injected retry queue, the native follow-up can
lose its internal-notification marker. The front recognizes that retry only
from its accepted local scheduling record, exact body, current session and
newer run generation, then checks the native completion receipt again. A
genuine inbound owner message always keeps its owner classification, even
when its words equal the pending review.

The watchdog polls every 100 ms, an assumption. At a tool boundary, a
completion review with no visible text receives the yield line after ten
seconds of tool work. Owner turns have no time limit, yield line or deadline
interrupt. A waiting owner can still interrupt an in-flight review request;
that review is retried after the owner turn, as described above. The front
schedules no deadline continuation. The yield line asks for the reply the
owner asked for, in the form asked, using finished tool results. Only
unfinished work goes to a hand. Hermes' own
"Operation interrupted." line is unchanged. The host must enable gateway injection. Missing bindings and
non-cooperating tools limit interruption; timeout-free tools retain native limits.

Front request rules run inside the existing hands request middleware, preserving
its audio and identity guards. Owner turns keep every caller-supplied token bound;
the front adds none. Reviews and system turns lower existing bounds to at most
512; tighter bounds survive provider fallback and `extra_body` overrides.
Missing bounds stay absent. Thinking uses the caller's setting or the route's
default. Tool calls are counted as reads or writes and never capped. The front
leaves tool arguments unchanged. It sends `action: block` with `message` for
every tool on a duplicate review, and for every tool except a few hand-offs when
yielding to a waiting owner.

The timeout inventory reads numeric `timeout` and `timeout_seconds` arguments
from registered tool schemas. There is no front timeout ceiling for `terminal`,
`process`, `browser_cdp` or `connections`: caller values stand, and each tool's
own default applies when no value is supplied. Background
terminal calls retain their own timeout. `execute_code`, `web_search` and
`web_extract` have no per-call timeout argument. MCP and plugin schemas without
one are listed in `FrontGuard.unbounded_tools`; interruption only reaches tools
that cooperate. No unknown timeout argument is invented.

The ingress ledger records real platform/chat/message identities, sender and
text hash. Structural turn history is observed before branch reactions, answer
reads or capture. Only a matching unconsumed inbound record grants owner
provenance. No runtime suffix is subtracted from owner input; transformed or
ambiguous owner input refuses capture without becoming a completion review.
A synthetic turn cannot answer an open ask or consume a pending branch verdict.
The bridge exports `owner_started_turn(session_id, turn_id)` for answer capture and
`origin_ref` for dispatch. CLI, API and voice are outside the messaging guard.

A review requires an exact match to an owned native completion receipt. The
front reads `async_delegations.event_json` through a read-only SQLite connection
and uses Hermes' own formatter, including its coalesced batch envelope. The
50 ms read budget and 256 recent receipt limit are assumptions. A missing or
ambiguous receipt leaves a system turn eligible to speak. All ready results
are rendered together within 900 characters, with at most twelve compact result
lines, an assumption; omitted results stay eligible. The successful native
settlement outcome, not the gateway settled hook, advances reviews. Controller
writes run off the native settlement path. Accepted render receipts are stored
atomically in the private spool directory and survive restart. Silence never
communicates an obligation; delivery stays unknown without a platform receipt.

Duplicate reviews end in the exact `[SILENT]` marker. Their bound turn disables
native streaming and interim callbacks, since `stream: false` in a request is
insufficient. The callbacks are restored for the matching turn. Tests observe
actual native draft, edit and send output, including delayed agent promotion.
Unrelated internal notifications remain visible.

A hand's approval notice waits until its parent completion review ends, whether
the review replies or settles silently, then waits the existing 5 second gap.
Earlier tool work does not change this ordering. The controller stores a fixed
deadline at the earlier of 3 minutes after the hold and the hand's wall deadline.
The body releases the notice at that deadline if the hand or review hangs or
the completion is missing. A restart without a review receipt uses that same
deadline. Owner-started asks retain their conversational pause exemption.
After the review and gap, delivery does not start another chat wait. The verified
owner request link remains with the ask if controller records are retired.

A child held for approval finishes its native turn with a local report that the
action has not run. It needs no further model request or provider slot to hand
the blocked result to the parent's completion review. Prior tool results remain
available to that review.

Integrator: `tests/hermes_adapter/test_hands_front_integration.py` is opt-in.
Set `COLONYMIND_HERMES_INTEGRATION=1` and `HERMES_SOURCE_ROOT` to the pinned
checkout, with Hermes dependencies available, then run that test file. It uses
a temporary profile, native adapters and stream consumers, and a real local
foreground terminal. It does not call a
live model or deployment. Run the full adapter suite where HTTP listening
sockets and an installed Hermes distribution are available.

The controller's `owner_message`, `review` and `communicate` methods are served
at `/v1/hands/owner_message`, `/v1/hands/obligations/{oid}/review` and
`/v1/hands/obligations/{oid}/communicate`, with the instance key and an
idempotency key like every other write. A refused controller review never
authorizes duplicate silence.

Hermes puts its own notes ahead of what the owner sent (a reply's quote, a voice
transcript, a media note), joined by a blank line. When the platform message id
binds the turn to the owner's inbound record, those notes keep the turn his: the
capture keeps the whole text, and the answer route and a branch verdict read the
words the adapter heard him send. Text after his words, or a recorded runtime
line left inside, cannot be split exactly and is refused.


## Evidence

`hands.brief.build_brief` preserves the goal, context, intent version and fixed
acceptance criteria. Its design assumptions are 6,000 tokens total, 2,500 for
attributed recall and 900 characters for identity. The default counts UTF-8
bytes conservatively; a host can supply its tokenizer. Required context that
does not fit is refused. Recall uses the parent's established owner standing
and the ordinary owner assembly; a non-owner never invokes that callback.

`CheckRunner.run` executes only the saved plan's command or probe, after work
stops, within the 300-second design assumption. Code receipts bind HEAD,
porcelain and file contents, including dirty files whose status is unchanged.
Artifacts must parse or decode. Research quotes must occur exactly in text
observed by this hand; source independence uses distinct hostnames as a
structural assumption, not proof of editorial independence. An operations
probe must match each named requirement as an exact output line. Merely
exiting zero establishes execution, not the desired state.
An ops hand stopped at a hold may have no planned probe or working directory.
Then the checker produces no receipt; it does not manufacture a `TypeError`
failure. A planned check without a working directory still fails closed.

`hands.evidence.assess` and `evaluate_result` use scoped runner receipts and
the recorded intent. E0 is self-report; E1 is observed execution; E2 validates
an artifact; E3 tests the requirement on its final binding; E4 adds independent
family review; E5 records an owner verdict. Only E3 and above may be called
verified, with the scope. Conflicting final bindings cannot retain a prior
success. A changed artifact requires another check. Executor observations,
review verdicts and owner verdicts must come from their trusted readers.
These callbacks do not authenticate a producer that can forge executor data.

`Controller._brief` uses `build_brief`; `hands.runtime.configure` supplies the
ordinary `_assemble_sections` owner view with the exact goal query and
`read_only=True`. Standing comes from retained session attribution in the turn
ledger, never task words. Missing, mixed or child-session standing gets no owner
recall. Recall failure leaves the fixed brief intact. The read-only assembly
records neither learning feedback nor a lesson use. Host identity and style text
come from `hands.persona_addendum` and `hands.style`. Identity substitutes `{name}`
and `<name>` with the controller's hand name; other host text is kept verbatim.
Identity, neutral guidance and style are joined with single spaces in that order
by the same renderer in the controller brief and request middleware.
Both settings default to empty strings; the controller brief always
includes neutral return and evidence instructions, with no default persona or
style rule. These passages together must fit the 900-character identity limit.

`Runtime._spawn` sends the returned controller brief as child context and keeps
the existing local brief in degraded mode. `Runtime._checks` uses the saved
intent's exact acceptance dictionary exposed in the hand's decisions, with a
saved-task lookup for older spools. `budgets._tool_receipts` retains observed
file paths and fetched text; summary prose supplies neither. A fetched page is
only what a proposed quote is checked against: no structured quote reaches the
runner yet (`proposed_quotes` is empty), so a research hand gets no runner
receipt and stays at E1 until its return contract carries quotes with their
URLs. `subagent_stop`
records checking before running and finalizing checks outside `DISPATCH_LOCK`.
Hermes retains the slot throughout the hook. Runner failures, including invalid
configuration and timeouts, become failed wire receipts. A stopped budget hand
still submits its partial result (see Budget stops).

After reconciliation, `Runtime.probe_unknown` reads the controller's saved plan
and runs `CheckRunner.probe`. Coding observes HEAD and porcelain; ops runs only
the saved probe. The authenticated `POST /v1/hands/{id}/probe` checks the original
executor epoch and unknown-effect state, records the observation on the hand,
and exposes it in the digest. No effect is retried. Runtime routes live in
`hands.runtime_router`, included alongside the shared contract router by the server.

Tests are `sidecar/tests/test_hands_brief*.py`,
`sidecar/tests/test_hands_evidence*.py` and
`tests/hermes_adapter/test_hands_checks*.py`. The real HTTP research test is explicitly opt-in:

```sh
COLONYMIND_TEST_HANDS_SOCKETS=1 PYTHONPATH=sidecar python -m pytest -q tests/hermes_adapter/test_hands_checks_socket.py tests/hermes_adapter/test_hands_checks_ops_socket.py
```

The tests fetch fixture text over a local socket, check both a present quote
and an absent one, and probe a service before and after its named state changes.
No test contacts a deployment. Run the full adapter suite
with installed stock Hermes after wiring the lifecycle calls.


## Provenance

`hands.provenance.Provenance` writes one owner-audience autobiography source
per accepted E3 or higher result. The source embeds the evidence scope and
receipts, so support survives branch expiry. Retries reuse its derivation;
E0 through E2 never create an autobiography entry.
An independently reviewed analysis can retain its scoped review receipt, and
an owner-accepted result can retain the attributed owner message and source
version, even when neither result has a command-runner receipt.

The digest renderer calls `Capture.record_hand_refs(session_id=..., turn_id=...,
digest=...)` after rendering the controller's digest. It records only complete
result lines actually shown, binds them to that exact turn and persists the
snapshot beside the assistant message. Model text and hook arguments cannot
upgrade it. `memory/recall.py` adds the marker to assistant quotations and
conversation pairs by reading the original message metadata. A relay of E0
work remains "own analysis, not checked", even if its words say "verified".
Assistant quotations continue to supply no source claims.

The lesson intake and witness gates read hand evidence and ancestry. An
unchecked relay cannot teach, including through the nightly intake. Supported
outcomes, stored failed checks and attributed owner verdicts can supply
lessons. Every result entry, hand lesson and relay has a derivation. Parent
erasure retracts the entries and lessons; ordinary branch expiry leaves
embedded receipts intact and marks relays "evidence expired". An owner
reversal disputes the retained entry and invalidates its lessons.

`Controller.result` and `accept` call `record_accepted` inside the original
idempotent mutation. Result text is hash-bound, retained in the controller's
result decisions for explicit acceptance, and removed on parent erasure. The
runtime obtains `get_turn_idempotency_ledger(get_state_dir())` and the owner
contact id. Missing dependencies or a provenance failure are logged without
failing a result. Replay never invokes acceptance again.

`FrontGuard._shown`, called by `_digest`, passes the exact rendered snapshot to
the live adapter's `record_hand_refs`, through the capture bridge module the
adapter registers in `sys.modules` at load and removes at unload.
Only complete rendered lines with controller evidence enter the snapshot.
Compact review lines carry their shown text. The front also submits the exact
hand IDs to authenticated, idempotent `POST /v1/hands/relay`. A turn's first
snapshot is immutable and bound to its session; hand sessions are refused.

Source forgetting appends curated tombstones and fences recorded source dependencies in recall;
it preserves parent sessions, controller ancestry, entries and lessons. The hands controller's
independent `Controller.erase_parent` lifecycle tombstones derived sources through the same curated
journal instead of deleting source rows. `BranchStore.prune`
retains durable expiry markers, including the child session, so pruning on a
read path cannot lose the provenance notification. Controller maintenance
drains them through `hands.runtime.expire_branches`, deleting each marker once
applied; without the turn ledger or an owner the markers wait.
Expiry keeps embedded accepted receipts and downgrades relay support.

The trusted mind reader `Lessons._vetted` passes a wrong verdict and its exact
preceding reply reference to `hands.runtime.owner_verdict`, which disputes only
hands in that reply's stored metadata. `hands.runtime.admit_hand_plan` is the
nightly intake hook; only a failed check the controller holds makes a cited hand
an observed failure. Direct and nightly hand lessons use
`Provenance.admit_lesson`; ordinary admission refuses hand evidence or lineage
without that derivation path. A lesson combining different hands waits for a
supported synthesis. These hooks write no external memory service.

Runtime integration regression tests are `sidecar/tests/test_hands_wave1_wiring.py`
and `tests/hermes_adapter/test_hands_runtime_wiring.py`. Scenario stimuli use the real runner and
production result path for restart and duplicate completion, and arm the real
review watchdog through `pre_llm_call` with an internal notification and a bound
agent generation for owner arrival. Their guard-off arms still detect the
removed guard. Socket and live Hermes checks remain explicitly opt-in.

`sidecar/tests/test_hands_provenance*.py` covers capture through ordinary
recall and lesson intake, embedded support, retries, expiry, erasure and
disputes against real source and controller stores. The async lifecycle tests
need local socket-pair wakeups in a normal test environment.


## Authority

The sidecar refuses memory writes, erasure, people changes and owner-answer
routes when a structured request identity names a registered hand session.
The request dependency reads `X-ColonyMind-Session`, `session`, `session_id`
and `requester`, including the turn context. Retired and rotated child ids
remain hand identities. Parent capture of a child's branch remains possible:
`child_session` names the captured evidence, not the requester.

`hands.sessions.register` validates the hand, parent and executor epoch;
the controller's existing `hands.child_session` binding is also recognized.
The adapter tripwire refuses key reads and answer-route calls from the hand
registry and reports `answer_forgery_attempt`. The owner's own conversation
retains its existing tools. Shell-capable hands, including mind-started hands,
remain trusted lanes with budgets, named-tool restrictions and best-effort
recognition. Unrecognized commands run. These checks are not a sandbox against
a caller that can read the instance key and forge or omit session identity.

The plugin uses the adapter's copy of the mind effect table for money,
irreversible deletion, credential changes and bulk messages. Publishing and
owner account access are recognized through `hands.floor.publish_targets`
and `hands.floor.owner_account_hosts`. Literal workspace-local deletion is
exempt; a symlink escape or a compound deletion is not. A recognized deletion
whose targets cannot be bound is refused. Files are hashed, git pushes bind
the source commit or ref set and resolved push destination, destinations are
canonicalized, and configured service/probe targets use a fresh host probe
result. No destination or account identity is built in.
Workspace checks use Hermes' actual terminal directory, including earlier
directory changes. A recognized executor effect with no reliable local working
directory is refused; an embedding host can supply `runtime.workdir_for`.

Holds reuse `Mind._hold` with origin `hand`. Their asks show only structured
call arguments, working directory and target bindings. Hand prose supplies
neither authority nor an ask's binding. Hand asks are notes so the legacy mind
dispatcher cannot also start a kanban worker for the same approved action.
Continuations belong to the hands controller and native plugin continuation
path. A continuation may consume an ancestor's permit for its current task
intent, never a sibling's or another task's permit.

An owner's yes resumes the holding hand with the workspace of its task's first
attempt, or with none when that task had none. The held call's directory is
where that one call runs, never a workspace grant, so a held `rm` in a broad
directory cannot exempt later deletions there. The continuation's exact
approved call always goes through the permit check and consumes the one-use
permit, even inside a workspace. Every other deletion or floor effect follows
the original workspace's rules and otherwise waits for its own approval.

An identical held call from the same parent session replaces older open asks
for that call digest, working directory and target values. Under the hands
write lock, their waiting or granted permits become stale and their asks are
cancelled before the newer hand creates its ask. The older hand and its
continuations cannot use those permits afterward. Approval never transfers:
the newer ask requires a fresh owner answer. Different sessions or different
bindings remain separate.

A native completion whose only news is a trusted approval hold settles
silently before model execution or streaming. The controller must report a
stopped, blocked hand and the executor must retain its awaiting-approval hold.
Mixed completions, artifacts, fetched text, verified results and other
observed work retain normal review. Only a successful plain `pwd` observation
and calls vetoed before execution add no result beyond the hold. Failed or
incomplete calls keep their review.
The silent result creates no turn or reply in the answer ledger, so the ask
remains last in RCS, WhatsApp and SMS. Owner-started asks still skip the
conversational pause. If an unrelated reply displaced an ask, an unbound clean
yes or no shows the sole recent ask again in that chat for a fresh answer.
The front must not start another hand to repeat it or direct the owner to
another channel. See [APPROVALS.md](APPROVALS.md).

`Approvals` reads approval exclusively from the mind's structural owner-answer
history, preserving the answer message reference and the configured permit
expiry. The answer route synchronizes grants after the answer is stored and
before returning the trusted hand, permit and exact call to the runtime.
The held-permit endpoint refreshes that state for runtime recovery after an
ingested answer or a lost response. Its underlying service read does not mutate
or consume permits. Without the mind, stored bindings remain readable but the
endpoint marks them unrefreshed, so the runtime cannot resume from them.
The check atomically consumes the permit in
SQLite immediately before execution. Expired, withdrawn, changed-intent and
already-consumed permits refuse. Changed target values mark a permit stale
and form a fresh ask. Hold requests have durable idempotency records. A check
never replays an `allow` response: losing that response requires a new approval
rather than risking a repeated effect. Binding immediately before dispatch
narrows the race; it does not prevent a concurrent filesystem write afterward.

When an owner's answer grants a hand's permit, the runtime automatically resumes
the owning hand exactly once. It waits for the current attempt to end before
starting its continuation. The continuation runs the held call through its exact,
single-use permit and reports through its normal completion. The answer turn's
context says the hand was resumed and the model must not continue it again or
perform the action itself. A duplicate continue is a no-op. Only a grant resumes
the hand; a no, a lapse or a refusal resumes nothing. The console's yes remains
refused and grants no authority.

A second owner channel (platform `companion`) can read the open hands permit
asks in full and answer one by its immutable id, through its relay's own
`answer` credential, never the instance key
([hands-permit-answer-contract.md](hands-permit-answer-contract.md)). Its
answer settles the same ask row through the same settle path and lock as an
RCS answer. The settling write is a compare-and-set in SQLite, so the first
commit wins and a later answer from either channel gets the recorded winner.
Each app answer is also told to the owner on RCS. The grant and the resume
above are unchanged.

In the owner's own session, a front turn confirmed as owner-started by the
ingress ledger can run recognized floor effects directly without a hand, hold
or approval. A missing ingress ledger retains this owner-session exemption.
The exception is a call matching a waiting or granted permit held by a hand of
the same parent session: the same bound targets or the same call digest. This
call is blocked because the runtime resumes the holding hand after approval.
A different recognized command on the same bound target
is still blocked, including in the owner's answer turn after approval.

Held targets are read from the controller or synced hand records, so this check
survives a plugin or gateway restart. If those targets cannot be read, owner
floor effects are blocked with a clear reason only while a hand of the same
session awaits approval. When no hand is waiting, unavailable held targets do
not block the owner's turn.

Front turns that are not owner-started, including completion reviews and
injected turns, cannot run recognized floor effects or borrow any hand's permit.
After an owner approves a held call, its holding hand's continuation checks and
consumes the exact permit before execution. Hand effects outside that permit
need their own hold and approval.
Stopped, waiting or already refused hand calls cannot consume a permit.
Completion of a held effect requires both a consumed permit and the permitted
hand call's successful effect receipt.

The sidecar refusal needs the request to name its session. The answer and
people-change bodies carry none, so the adapter sends `X-ColonyMind-Session`
from every tool call and from the answer read (`client.acting`); a caller with
the instance key and a full shell can still omit it, so the plugin's
hand-session refusals (allow-lists, the named-tool blocks and the tripwire)
remain the enforcement there.

The controller populates the hand and session rows and drives
continuations. The shared controller path is `hands.db` in the instance directory;
each guarded request opens it for one lookup once the file exists. An optional
`runtime.probe_target(kind, target)` supplies fresh bytes for service or probe
bindings; a missing required probe refuses execution. Authority tests live in
`sidecar/tests/test_hands_authority*.py` and `test_hands_permits.py`.
Set `COLONYMIND_TEST_HANDS_SOCKETS=1` to include the
isolated real-socket refusal test. No test contacts a deployment.
