# Persistent Hands

A Hand definition saves a name, role, instructions, avatar, tool and connector
allowlists, memory entries and skills. It supplies context and restrictions to
the existing Hands controller and Hermes delegation runtime. It grants no owner
standing, permit or approval evidence.

The API uses the instance credential with owner standing, or an issued credential
whose sole access operation is `hands`. Registered child sessions are refused on
the definition, pending-request and transcript routes. All issued `hands`
credentials receive 403 outside `/v1/hands/*`, including permit-ask answers.

```sh
colonymind credentials add companion-hands --access hands --owner-platform companion --contacts own --key-file /private/companion-hands.key
```

The Companion hub must use the owner-platform grant above to start tasks.
`hands` remains its only access operation. The credential can manage definitions,
queue owner-started tasks, inspect Hands and perform the owner actions below,
including answering a Hand's question. It cannot answer permit asks, call the
internal claim/bind/release routes, or reach any other API area.
Without an owner platform, a hands key can submit requests for inspection; they
remain `awaiting_owner_start` and appear in `pending`, without automatic execution.
Enqueue stores the authenticated initiating principal, credential and
`owner_started` flag. Request bodies cannot supply that authority. The instance
key and authenticated credentials that speak for the owner also establish an
owner start, following the live owner-goals authority model.

All mutations require a nonblank `Idempotency-Key`. Repeating a key replays its
first committed response; changing the body returns 409 `idempotency_key_reuse`.
Use a fresh key for a new mutation. Definition names
are trimmed, case-sensitive and unique among definitions that are not archived.

| Method and path | Contract |
| --- | --- |
| `GET /v1/hands/definitions` | `{definitions: [...]}`; `enabled_only` and `include_archived` boolean filters |
| `POST /v1/hands/definitions` | Create from `{name, role?, instructions?, avatar?, tools?, connectors?, enabled?, provenance?, memory?, skills?}` |
| `GET /v1/hands/definitions/{id}` | Full current definition, including archived definitions |
| `PATCH /v1/hands/definitions/{id}` | Replace the whole editable set; optional `expected_version` returns 409 on mismatch |
| `DELETE /v1/hands/definitions/{id}` | Archive, disable and increment the version; keep history, tasks and permits |
| `GET /v1/hands/definitions/{id}/tasks` | `{tasks, limit, offset}`, newest first; limit 1–200, default 50 |
| `POST /v1/hands/definitions/{id}/tasks` | `{brief}` queues a task and snapshots the definition immediately |
| `GET /v1/hands/tasks/{id}/messages` | `{messages, available}` from existing redacted, erasure-aware child branch readers |

The Companion credential has an explicit **method + route template allowlist**
in the Hands router guard. Its complete surface is the definitions CRUD and
definition task create/list routes above, plus:

| Method and path | Request and response |
| --- | --- |
| `GET /v1/hands` (redirects to canonical `/v1/hands/`) | Optional query filters `session`, `task_id`, `state`, `kind`, `priority`, `origin`, plus `cursor` (default empty); `{hands, cursor}` |
| `GET /v1/hands/{id}` | Execution attempt (Hand) |
| `GET /v1/hands/tasks/{tid}` | TaskView, including questions and status below |
| `GET /v1/hands/tasks/{tid}/messages` | `{messages, available}` |
| `GET /v1/hands/digest?session=...&cursor=...` | `{text, items, cursor}`; `session` required, `cursor` defaults empty |
| `POST /v1/hands/questions/{qid}/answer` | `{"answer":"<text>"}` → question with `state: "answered"` |
| `POST /v1/hands/tasks/{tid}/revise` | `{"goal"?,"criteria"?,"stop"?}` → updated TaskView; goal or criteria required, stop defaults false |
| `POST /v1/hands/tasks/{tid}/cancel` | `{"reason":"<text>","owner_ref"?}` → TaskView with raw `state: "cancelled"`; owner_ref is optional even for a task with an obligation |
| `POST /v1/hands/tasks/{tid}/accept` | `{"reason":"<text>","hand"?,"for_version"?}` → TaskView with raw `state: "done"` |

Every other mounted Hands route returns 403 `hands_route_not_allowed` for this
Companion credential. This includes `/plan`, `/owner_message`,
`/{id}/dispatched`, `/release`, `/bind`, `/events`, `/question`, `/hold`, `/result`,
`/stop`, `/name`, `/probe`, `/permits/*`, `/obligations/*`, `/notices/*`,
`/reconcile`, `/relay`, `/retention`, `/requests/pending` and all definition queue
claim/bind/release routes. New route templates and methods are denied by default.
`GET /v1/hands/requests/pending` remains a main-agent/runtime view (up to 100
pending requests, oldest first), accessible with the existing instance credential.
A narrow hands credential without the `companion` owner platform still cannot
answer Hand questions (403); its task requests still await an owner start.

For answers, `answered_by` is optional for existing callers. The Companion
server attribution always comes from the verified credential's identity
(`credential:companion-hands` for the command above); a submitted `answered_by`
never changes attribution. Existing instance callers may continue supplying it.
The response is the question itself: `id`, `hand_id`, `asked_at`, `text`,
`assumption`, `blocking`, `state`, `answer`, `answered_by`, `answered_at`.
Answering a question grants no permit and does not answer a permit ask.

For accept, omitted `hand` selects the latest attempt for the task's current
intent version; omitted `for_version` uses that version. A supplied version
different from the current task intent returns 409 `stale_result`. The selected
attempt must have result evidence and state `done`, `below_minimum` or
`superseded`, and the task must not be cancelled; otherwise 409
`acceptance_guard`. An in-flight latest attempt cannot cause an older success
to be silently accepted. Existing callers supplying both fields retain explicit
judgment of a superseded result for the current intent version.

`criteria` keeps the existing acceptance-criteria object (including
`test_command`, `probe`, `required_sources`, `independent_sources`,
`quotes_required`, `output_type`, `requirements`). `owner_ref`, when supplied,
is `{platform, chat, message_id}`. All owner actions require `Idempotency-Key`;
identical retries replay the saved response even after state advances. Changed
bodies, including a changed ignored `answered_by`, return 409
`idempotency_key_reuse`. Missing, empty or blank keys and invalid bodies return
422. A new answer operation on any non-open question returns 409
`question_closed`; revising a cancelled task returns 409 `task_closed`.
Unknown task, Hand and question IDs return 404 (`unknown_tasks`, `unknown_hands`,
`unknown_questions`). No owner action invents an owner message reference.

Create accepts defaults: empty role/instructions/lists, enabled true and
provenance `owner`. An empty tool or connector list allows none. PATCH requires
name, role, instructions, avatar, tools, connectors, enabled, provenance, memory
and skills. Limits are 80 characters for a
name, 200 for role, 8,000 for instructions, 200 memory entries of up to 4,000
characters each, and 50 skills of up to 20,000 characters each. Entry IDs must be
unique within their respective lists. All stored changes, including archival,
create immutable version snapshots in the existing private `hands.db`.

Avatar writes accept only colors `#B9A7D8`, `#9EBBA0`, `#D8B99A`, `#A0B9D0`
and shapes `circle`, `flower`, `square`, `star`, with exact case and spelling.
The defaults (including an omitted avatar or `{}`) are `#B9A7D8` and `circle`.
Explicit null or any other color/shape returns 422 on create and PATCH.
`ref` is optional/null, at most 2,000 characters, and unused. Existing stored
avatars and immutable snapshots remain readable without revalidation or rewrite;
new edits must use the enum. Definition errors are 404 `unknown_hand_definition`,
409 `definition_name_in_use`, `definition_version_mismatch`,
`archived_hand_definition` or `disabled_hand_definition` as applicable.

The main agent can use `colonymind_hands` with `action: "list_hands"` to see enabled
definitions as `{id, name, role}`. `delegate_task` accepts `hand` on each task.
It resolves a definition ID or exact name at the
controller. Unknown, disabled and archived definitions are refused before any
worker is dispatched. Ad hoc delegation retains its existing policy.

```json
{"tasks":[{"goal":"Review these sources","kind":"research","hand":"Source reviewer"}]}
```

Owner-started HTTP task creation reports `dispatch_state: "queued"`. While the gateway is
running, the existing Hands maintenance loop claims requests and dispatches them
through `Runtime.dispatch_tasks`, using the most recently captured authorized
live direct owner main session. Group, shared, guest, mind and system sessions
cannot serve as pickup parents. The current object session ID must match its
selected identity; compression causes recapture and revalidation under the new
ID. Before consuming the binding and again at native launch, after bind and
dispatch-intent persistence, the runtime rechecks current
owner standing, live sender authority (including configured owner handles),
chat type and the same captured destination. The loop waits at most two seconds between passes,
including when `hands.spool.replay_seconds` is larger. Capturing an owner parent
wakes the same loop early. Each pass can pick up to the configured running capacity, trying older refusals
after requests that have not been checked recently. No model action is needed.

The runtime uses its existing instance credential (`api.key`) for
`POST /v1/hands/definitions/queue/claim`, `/queue/bind` and `/queue/release`.
These internal routes refuse child sessions and the narrow `hands` credential.
Claims last 30 seconds. A single-use binding immediately before native dispatch
fences expired claimants and cached plans across gateway processes. An unbound
claim expires back to the queue; a consumed binding cannot be claimed again.
An uncertain outcome after binding follows existing reconciliation controls.
Authority lost after binding refuses the attempt visibly while keeping the
binding consumed. Owner controls can inspect or cancel it; polling cannot
execute it again. Native construction and lifecycle hooks share the checked
parent identity even if compression subsequently rotates the live parent.

Without a live authorized parent the task stays queued with
`waiting_for_main_session`. Admission refusal records `refused:<code>` and
retries on later passes. Unstarted retries consume neither attempt nor cascade
budgets; the attempt is charged when its one-use binding succeeds. Disabled or archived definitions are checked both at
claim and immediately before dispatch. Dispatch retains the existing platform,
detached completion consumer, route, reservation, child binding and result,
review and notices path. It creates a result obligation for the captured return
reference without recording an owner message or granting permit authority.
Manual `colonymind_hands` actions `pending` and
`{"action":"dispatch","task_id":"t-...","kind":"research"}` remain available;
manual pickup uses the same claims (`kind` defaults to `deep`).

A task and its continuations retain the definition version captured at task
creation. Later edits apply to new tasks. A task's worker brief includes the
saved name, role and instructions, then memory and skills, alongside the shared
persona and authorized shared recall. `imported_untrusted` definition text is
explicitly marked as untrusted data, never instructions. Existing brief and
route context limits still apply; an oversized required snapshot is refused
with `brief_too_large`, without truncation or dispatch.

The plugin resolves known toolsets into tool names at admission and freezes the
result for that attempt. Unknown names remain exact allowlist entries. Both the
existing kind policy and the definition policy must allow a call. Connector
names additionally restrict registered MCP and hosted connector identities.
Dispatch bridges are refused for definitions to prevent wrapping an otherwise
disallowed tool. Existing shell-capable kinds retain the trusted-shell boundary;
these named-tool policies do not provide an operating-system sandbox. Permits
still require the existing stored owner answer and exact-call checks.

Task views add `status`, `status_detail`, `definition_id`, `definition_version`
and `dispatch_state`; existing raw task fields remain. They also add `questions`
to both `GET /v1/hands/tasks/{tid}` and every task in the definition task list:

```json
{"questions":[{"id":"q-...","text":"Which source?","assumption":"Use the saved one","blocking":true,"asked_at":"2026-10-17T12:00:00Z","state":"open","answer":null}]}
```

Questions come from the questions table joined through the task's execution
attempts, never another task. At most 20 are returned, open questions first,
newest first within each group (question ID breaks timestamp ties). Erased
question text/answers remain null. A blocking open question projects
`needs_owner`, including after a task revision; an answered question's old `awaiting_input`
activity does not keep the task in that status. Status is `running`,
`needs_owner`, `review`, `done`, `failed` or `cancelled`. Queued tasks use
`running` with the queue detail; `status_detail` preserves the attempt state,
activity and outcome. Unreviewed results use `review`; accepted, reviewed
completion uses `done`. Existing attempt list/detail, digest and notices routes
retain their contracts.

Each committed change is published as `hands.changed` on the existing host
broadcaster and durable journal. Its payload has exactly these keys:

```json
{"seq":123,"changed_at":"2026-10-17T12:00:00Z","definition_id":"hd-...","task_id":null,"routine_id":null,"change":"definition.created","state":"enabled","version":1}
```

Changes are `definition.created`, `definition.updated`, `definition.archived`,
`task.created` and `task.state`. Payloads contain no names or text. `seq` is the
host journal's durable instance sequence. Version is the definition version,
or the task's saved definition version; it is null for ad hoc tasks. Routine ID
is reserved and null. A transactional outbox retries publication after failure
or restart, reusing the journal's immutable event key so a crash after publish
does not allocate a duplicate event.

The forward-only schema migration advances `hands.db` from version 1, 2, 3 or 4 to 5 and
adds definition, version, task-binding, request-queue, dispatch-claim and outbox tables.
Version 4 adds initiating authority fields and partial indexes for pending work,
claim expiry and unpublished/unprepared outbox rows. Existing queued requests
have no owner-start evidence and become `awaiting_owner_start`; their old claims
are revoked. Inspect and create a new owner-started request where needed.
Version 5 separates queue eligibility from initiating authority and consumed
bindings. Cancellation, terminal task states, revisions and stopped/superseded
attempts remove requests from the claimable index and revoke pending claims.
Admission refusals and lease expiry remain retryable. Migration applies the
same exclusions to retained rows without changing their owner evidence.
An empty claim does indexed reads and skips the writer transaction and event
outbox maintenance; work does not grow with retained completed or cancelled tasks or events.
Version checks and idempotent DDL execute under one `BEGIN IMMEDIATE` lock.
Stop old writers before migrating: older releases reject the newer schema. Existing
execution tables are preserved. Definition-bound task and permit history is
excluded from ordinary settled-task pruning, including after archival. Existing
source erasure remains in force for transcripts.

For definition mutations and task creation, reusing an Idempotency-Key with a different request body returns 409
`idempotency_key_reuse`; an unchanged retry replays its saved response.
