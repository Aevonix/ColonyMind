# The public API: the routes a client can rely on

The sidecar serves one HTTP API on loopback (port 7777 by default). Most of its routes are the Hermes adapter's: the
mind's pull protocol, the guard, the reports a body sends back. They change whenever the adapter does. The routes
below are the public subset: capture, context and recall, forgetting and corrections, what memory holds, and the
mind's reads. They are what a client of ColonyMind other than the adapter builds on.

## The spec

- **Each release** attaches `openapi.json` to its GitHub Release page: the OpenAPI 3.1 spec of the public subset,
  exported from the sidecar that release ships, carrying its version.
- **From an installed sidecar**, `python -m colonymind.api.public_spec [PATH]` writes the same spec (default
  `openapi.json`).
- **Every route**, public or not, is in the whole spec: `colonymind generate-types`, or `/openapi.json` on a running
  sidecar. A route that is not in the public subset may change in any release.

A public route keeps its method, its path and the fields it takes and returns within a major version. A release may
add a public route, an optional request field or a response field, so a client ignores fields it does not know.
The repository pins the subset's contract (every route, parameter and field, with its type, default, allowed
values and limits such as a maximum length or a page size, and the status codes and headers of its responses) in
`sidecar/tests/fixtures/public-api-contract.json`; a change to it fails the tests until it is reviewed, and the
changelog names it.

The mind's reads return JSON objects the spec does not describe field by field;
[the mind's API](MIND.md#the-api-v1mind-one-bearer-key) lists their fields.

### Staged voice context card

The staged endpoint `/v1/host/context/card` uses `GET` and is outside the frozen public subset. It requires an authenticated
credential with owner authority and `read` access (`context:read` for legacy scoped credentials). A query-string
owner identity cannot grant access. Guests, observation-only credentials, and anonymous development callers
receive no card bytes, version, or ETag.

`200` returns `owner_id`, `version`, exact `text`, `token_count`, `token_estimator`, `local_day`, `built_at`,
`checked_at`, `stale`, and `readiness: "ready_for_prewarm"`. Headers include `ETag`, `X-Context-Card-Version`,
`X-Context-Card-Stale`, `Cache-Control: private, no-cache`, and `Vary: Authorization`. `If-None-Match` supports
conditional `304` responses without a body. The weak ETag identifies the card bytes and owner binding; reconciliation
metadata can advance without a new card version. Revalidate using the same owner credential on every fetch.

`503` with `detail.code: "context_card_pending"`, `Retry-After: 5`, and `Cache-Control: no-store` means publication
is disabled, no card is ready, or an input revision awaits background reconciliation. GET never builds a card,
assembles turn context, or calls a model. See [the context card lifecycle and Companion handoff](CONTEXT-CARD.md).

Both specs carry the sidecar's own version (`info.version`), the release of the package that serves them.

## Refusals

A refused request answers `422` in one of two shapes, and the spec declares both wherever a route declares `422`:

- **An invalid field.** The request did not match the route's schema (a missing field, a wrong type, a value past a
  limit): `detail` is a list, one entry per field, with its location (`loc`), the reason in words (`msg`) and its
  kind (`type`) (`HTTPValidationError`).
- **The route's own reason.** The request matched the schema, and the route or a store it asked refused it
  (`Refusal`): `detail` is either an object whose `code` names the reason (`invalid_turn_id`, `invalid_page`,
  `invalid_request` for a value a store refused) and whose `message` usually says it in words, or the reason's code
  alone as a string (`transport_direction_status_mismatch`; a few older routes say it in words). Capture's `409` and
  `503` carry the object.

A client tells them apart by `detail`'s type (a list, an object or a string), and acts on a code (`detail.code`, or
`detail` itself when it is a string), never on the words.

## Credentials

Every public route except health needs a credential: `Authorization: Bearer <secret>` or `X-API-Key: <secret>`.
What a credential reaches is its binding ([Credentials](CREDENTIALS.md)): the `write` operation reaches the capture
routes, `read` reaches assemble, search, the inventory and a claim's history, and every other route is `manage`,
which by default only a credential granted the owner has. The mind's reads show the owner's view only to a
credential granted the owner. Without any key the sidecar serves loopback callers only, and never as the owner.

## The routes

### Voice recall (staged adapter route)

`POST` `/v1/host/memory/voice-recall` adds local keyword recall with `k=3–5`, a
hard `max_tokens=300–600` content budget measured by the repository estimator,
and partial/final utterance supersession. It uses search's `read` credentials
and canonical person/session scope. It is outside the stable public subset;
[the voice contract and local qualification method](VOICE-RECALL.md) describe
request IDs, cancellation, evidence formatting and process-local limits.

### Working-memory branches (adapter routes)

These routes support the Hermes adapter and are outside the stable public subset.
Each requires an owner-capable credential, an explicit `requester` equal to the
owner's configured person id, and the session's trusted sender. JSON requests
carry `sender: {platform, user_id}`; the listing uses `platform` and `user_id`
query parameters. The sender must resolve to the owner and its platform must be
granted to the credential. The local `cli` platform permits an empty `user_id`
only under its owner-platform grant. A guest's session cannot select an owner
branch by claiming `requester: "owner"`.

- `POST` `/v1/memory/branches`: `{parent_session, child_session, messages,
  brief?, summary?, requester, sender}` captures the completed worker's material
  and returns `{branch}`. Its spans are redacted ledger records, outside recall
  and claim formation.
- `GET` `/v1/memory/branches`: `session_id`, `requester`, `platform`, `user_id`
  list that parent's branches as `{branches, enabled, available, tool_available}`.
  With the faculty off the list is empty. `available` and `tool_available` state
  whether consults can run. An empty `session_id` is an availability probe that
  always returns an empty branch list.
- `POST` `/v1/memory/branches/session`: `{session_id, requester, sender}` builds
  `{branch}` from the session's existing captured turn records before compression.
  `branch` is null when the session has no retained records.
- `POST` `/v1/memory/branches/consult`: `{handles, question, session_id, requester,
  sender}` reads all named branches in one call. It returns `{answer, handles,
  ref, sampled, reference_answer, rating_format}`; the last two fields are null
  when no summary comparison was produced. Tool arguments are not accepted by
  this route.
- `POST` `/v1/memory/branches/reaction`: `{session_id, text, turn_id?, requester,
  sender, reference_text?}` admits the owner's next-turn words to the existing
  reaction reader. `reference_text`, when present, is separate explicit owner
  evidence about the summary answer; it is never inferred from a reaction to the
  served answer. The adapter can also pass the owner's next-turn JSON object
  unchanged as `text`, with exactly `answer` and `summary` fields holding their
  separate reactions, for example `{"answer":"Not helpful.","summary":"Keep going."}`.
  These are explicit owner preferences, not model-generated correctness scores.

Authentication or sender mismatches return `403`; malformed envelopes return
`422`. Disabled, withdrawn and summary-only consults return `409`; a missing
branch returns `404`, an input-size refusal `413`, and a spent hourly budget
`429`. A source changed during a consult returns `409`; an unexpected model tool
call returns `502`; an unconfigured consult router returns `503`. Store refusals
carry `detail.code` and `detail.handles`. Limits, expiry,
erasure and the comparator are described in [Working memory](WORKING-MEMORY.md).

### Capture

- `POST /v1/host/turns/sync`: one turn (the message, the reply, the sender and the session). Memory, claims and
  commitments are formed from it, attributed as the credential allows.
- `PUT /v2/host/turns/{turn_id}`: the same, idempotent by the turn's id: `201` the first time, `200` for an
  identical replay, `409` for changed content under the same id.
- `PUT /v2/host/turns/source-media/audio/{turn_id}`, `PUT /v2/host/turns/source-media/video/{turn_id}` and
  `PUT /v2/host/turns/source-media/document/{turn_id}`: a turn that carries that media. An older sidecar that could
  not keep the media refuses the path instead of dropping it ([audio](SOURCE-AUDIO.md), [video](SOURCE-VIDEOS.md),
  [documents](SOURCE-DOCUMENTS.md)).
- `PUT /v2/host/conversations/observed/{conversation_id}`: a conversation a wearable or room microphone heard
  around the owner, sent once and complete, with its speaker-labelled segments. It is not a turn: only the words
  labelled `owner` are the owner's, and it is never something said to the agent. Idempotent by the pair
  (`source`, `conversation_id`) like a turn (`201`, `200`, `409`), with the same `Idempotency-Status`, `202` and
  `503` answers. Only a credential
  granted the owner sends one (`403` `not_owner`); a conversation past the size bounds is `413`
  `observed_conversation_too_large`. `GET /v2/host/conversations/observed/{conversation_id}` requires a
  `source=<source>` query parameter and returns what memory holds of it, to the owner only. The source ID is
  `observed-conversation:<source>:<conversation_id>`;
  all speakers' words stay owner-side only, and `speaker_source_ids` is always empty. Conversation IDs allow
  letters, digits, `_`, `.`, `:`, and `-`, with a letter or digit first; slash is `422`
  `invalid_conversation_id`. All turn-writing routes refuse the `observed-conversation:` ID prefix with
  `422` `reserved_turn_id` ([observed conversations](SOURCE-CONVERSATIONS.md)).
- Every answer that takes a turn in says what became of it in the `Idempotency-Status` header (`created`,
  `replayed`, `erased`, `unkeyed` for a turn sent without an id, or `in_progress`). While the same turn is still
  being taken in, the answer is `202` with `Retry-After`: send it again after that many seconds. `503` means the
  turn cannot be taken in now: after an interrupted first attempt (inspect before sending it again), or with
  `Retry-After` while the sealed store is unavailable. A refusal carries a `code` in its `detail`.

### Verified work

Work records show what an admitted owner request caused and which postconditions
were verified. These routes require the owner instance credential (`api.key`).
Issued and legacy scoped credentials cannot read these records or submit late
work events, even when granted an owner platform for ordinary turn capture.

- `GET /v1/host/work`: roots, newest first. `limit` defaults to 50 and accepts
  1 through 100; `offset` defaults to zero and accepts 0 through 10000. Follow
  `next_offset` with the same `since`, `until` and `status` filters. `status`
  accepts `proven`, `partial`, `unproven` or `failed`.
- `GET /v1/host/work/{id}`: a root with its operation statuses and child lineage.
  Use the opaque `id` returned by list or detail, up to 1370 characters. The
  separate `turn_id` retains the original host identifier. Opaque IDs keep
  reserved names, slashes and Unicode source identities reachable by one route.
  Each effect includes its verification reason and source revision references.
  Unknown and erased roots return 404. At most 256 effects and children are
  returned; explicit truncation flags identify any omitted rows.
- `GET /v1/host/work/summary`: counts and rates by root status, roots per day,
  the separate owner-corrected rate, and up to 100 unknown effect kinds with a
  truncation flag. `overlays` counts `confirmed`, `corrected` and `none` separately
  from execution status. `titles.unproven` and `titles.failed` each contain up to
  five current titles, newest first; `untitled` counts rows without a title.
  `verdict_coverage` reports the latest successful lesson scan, its six-session
  and eight-turn limits, omissions and link counts; `reviewed_at: null` means no
  successful scan yet. These are the inputs for the external morning digest.
- `POST /v1/host/work/events`: immutable late child or hand facts using the
  versioned work event envelope. Its `root_turn_id` is the original host turn
  identifier, not the opaque read ID. `event_id` is the idempotency key. An identical
  retry is accepted with `duplicate: true`; conflicting content returns 409.
  A missing admitted root or child lineage returns 404 and can be retried after
  its parent arrives. Events cannot invent a root.

List and summary accept timezone-aware ISO 8601 `since` and `until` times.
Times select when the root request was admitted; its current late effects are
included with that root. The start is inclusive, the end exclusive, and a window
spans at most 366 days.
The end cannot be in the future. Without bounds, the window covers the last
seven days. Responses return the resolved bounds; keep those bounds while
paging. Newly admitted or erased records can change later pages.

Both turn capture versions accept the optional `work` envelope. When one is
present, `work_recorded` confirms its admission; `work_skipped_reason` explains
disabled or ineligible capture. An older server without that acknowledgement
cannot silently discard work on downgrade. Secrets are sealed before evidence
hashing and projection. A response contains safe
references and statuses, without raw arguments or tool results. With
`mind.faculties.work_records: false`, list and summary return empty data and
`enabled: false`, detail returns 404, and event admission returns
`accepted: false` with `skipped_reason: work_records_disabled`. No projection is
opened. Capture and verification run with the mind or lessons off and without
a model. Other hosts may use the same evidence contract. See [Verified
work](WORK.md) for lineage, verification and erasure rules.

### Context and recall

- `POST /v1/host/context/assemble`: the context sections for the reply to an incoming message, fitted to the
  characters the host puts inline; with `trace` (a credential that manages the instance), why recall kept or left
  out each candidate.
- `POST /v1/host/memory/search`: recall of one person's memory, with an optional date `window` and `as_of` time;
  with `trace`, why each candidate was kept or left out ([hybrid recall](RECALL-HYBRID.md)).

### Curated memory

- `POST /v1/host/memory/curated`: remember an explicit fact or preference independently of chat history.
- `GET /v1/host/memory/curated/search`: search current retained content with a query and cursor.
- `GET /v1/host/memory/curated/{memory_id}`: read current content and revision.
- `POST /v1/host/memory/curated/{memory_id}/correct`: append an override using the expected revision.
- `POST /v1/host/memory/curated/{memory_id}/forget`: append a tombstone using the expected revision.
- `GET /v1/host/memory/curated/{memory_id}/relationships`: read participant and recorded subject references.

Correct and forget return durable receipts (`id`, `revision`, `applied_at`, `operation`, `request_id`).
Retries with the same request id and payload return the original receipt. Canonical recall and search honor
these overrides and tombstones; historical source rows remain intact. All forgetting uses this append-only
journal. Chat deletion does not forget memory; forget does not delete chat. Curated reads require explicit person authority and the existing memory read/search/write scopes.

### Forgetting and corrections

- `POST /v1/host/memory/curated/sources/forget`: tombstone source evidence and its recorded recall dependencies
  ([source forgetting](RELATIONSHIPS.md#source-erasure)). Send `person_id` and either `source_ids` (1–100 ids)
  or `old_text` with `session_id`. Exact text must match a whole message of exactly one source in that session.
  Responses contain `source_erased: true` (compatibility: tombstone recorded), `source_ids`, `receipt`,
  `scope: curated_source_tombstone` and an explanatory `note`. Historical rows and external databases remain
  intact. Normalized repeated selections return the same receipt. The retired destructive URL returns
  HTTP 410 `memory_forget_moved` without writing anything; deploy the sidecar with or before the adapter.
- `POST /v1/host/memory/sources/annotations`: an attributed correction beside a source revision
  ([corrections](SOURCE-ANNOTATIONS.md)).

### What memory holds

- `GET /v1/host/memory/inventory`: the claims memory holds about one person, newest first, with their states.
- `GET /v1/host/memory/inventory/claims/{claim_id}/history`: one claim's chain: what it replaced, what replaced or
  withdrew it, the owner's answers and the corrections of it.
- `POST /v1/host/claims/dry-run`: what a message would become in a person's memory, with nothing written
  ([claims](SOURCE-CLAIMS.md)).

### Commitments and contacts

- `GET /v1/host/commitments` and `GET /v1/host/commitments/{commitment_id}`: the commitments captured from
  conversation, and one of them.
- `GET /v1/host/contacts/resolve`: the contact behind a handle on a gateway; with `create=true`, an unknown
  sender becomes a new contact, which link proposals reconcile later.

### The mind, read only

- `GET /v1/mind/state`, `GET /v1/mind/stats`, `GET /v1/mind/concerns` and `GET /v1/mind/goals`: the switches,
  drives, concerns and open goals, and the counts.
- `GET /v1/mind/log`, `GET /v1/mind/log/{intention_id}` and `GET /v1/mind/why/{intention_id}`: what the mind did
  and why.
- `GET /v1/mind/asks` and `GET /v1/mind/asks/stats`: the open asks, and how asks were answered.
- `GET /v1/mind/grants` and `GET /v1/mind/allows`: the standing grants and allows in force.
- `GET /v1/mind/owner-gate`: the owner gate's audit ([the owner gate](OWNER-GATE.md)).
- `GET /v1/mind/narrative`, `GET /v1/mind/drift`, `GET /v1/mind/lessons`, `GET /v1/mind/wishes` and
  `GET /v1/mind/learning`: the self-narrative, the identity check, lessons, wishes, and what the learning rules
  changed with the preference changes held for the owner.
- `GET /v1/mind/opinions` and `GET /v1/mind/opinions/{opinion_id}`: the agent's opinions ([opinions](OPINIONS.md)).
- `GET /v1/mind/people`, `GET /v1/mind/people/proposals` and `GET /v1/mind/people/{reference}`: the people the
  agent knows, the handle links waiting for the owner's decision, and one person ([people](RELATIONSHIPS.md)).

### Outside observations

Every observation route requires an owner-capable credential. A guest or
contact credential cannot read or post outside observations. An optional
`viewer` narrows a read; it never widens the credential's audience. A hub's
`observe` credential reaches only the post and cap routes, and only for the
sources it lists (403 `source_not_granted` for any other); see
[CREDENTIALS.md](CREDENTIALS.md).

- `POST /v1/mind/observations/sources/{source}` replaces only the named
  source's set. The body names the same `source` and an `observations` list;
  an empty list clears that set. It leaves the body's board observations
  and every other outside source intact.
- `GET /v1/mind/observations` returns the owner's active outside sets and
  report counts for the past day per source: surprises, concerns, confirmed
  message deliveries and ignored observations. Expired or missing sets
  contribute no active observations. Surprises use capture time, concerns
  and ignored observations use receipt time, and messages use delivery time.
- `GET /v1/mind/observations/sources/{source}/cap` returns
  `{source, cap, basis}`: the source's daily cap and where it comes from,
  `runtime` (set by the owner, with `set_by` and `set_at`), `config`
  (`mind.observations.source_caps`) or `default` (`daily_source_cap`).
- `PUT /v1/mind/observations/sources/{source}/cap` with `{"cap": n}` (a whole
  number from 0 to 1000) sets the owner's runtime cap for that source, and
  `{"cap": null}` clears it. It survives a restart, stands over the
  configuration until cleared, and each change is a note in the mind's audit
  log (`observation_cap`). It returns the same shape as the read.

For example, the body for source `device-a` is:

```json
{
  "source": "device-a",
  "observations": [
    {
      "key": "sample-1",
      "kind": "numeric_surprise",
      "subject": "device-a",
      "feature": "temperature",
      "observed_at": "2026-09-20T12:00:00Z",
      "severity": 0.5,
      "summary": "The reading exceeded its expected range.",
      "expires_at": "2026-09-20T14:00:00Z"
    }
  ]
}
```

`kind` is `numeric_surprise` or `percept`. `observed_at` is the observation's
own UTC capture time in ISO 8601, never its relay or receipt time; severity
is a finite, non-negative number. `summary` is short and stored as
`derived_unverified`. `expires_at` is optional and can shorten the default
expiry, not extend it. Defaults are assumptions: expiry is two hours after
capture and the persisted daily cap is six observations per source per UTC
day. Stale or missing sources contribute no active observations and raise
no new concern, error or wait. Stored counters have a 30-day horizon
(implementation assumption), with older counters removed on that source's
next post. Expiry does not erase original images or close existing concerns. With
`mind.faculties.senses: false`, posts are accepted and ignored.

A percept may also carry `image`, a PNG, JPEG or WebP data URL. An image
requires `device_principal` and `frame_hash`, the SHA-256 digest of the
decoded bytes. Its size cap is 512 KiB (assumption), reducible through
`mind.observations.image_max_bytes`. The canonical image ledger stores the
original under source kind `observation` with its capture time and owner
audience. Without an original, a percept can only become a note.

Numeric surprises are upkeep notes only and never trigger tasks, messages,
holds or operational changes. Percept routing is selected locally through
`mind.observations.routes` (`ignore`, `note`, `look` or `tell` per source),
not through a posted action field. Looking reopens the original through the
worker path; telling still passes the owner gate and existing budgets. See
[Observations and senses](MIND.md#observations-and-senses).

### Health

- `GET /v1/host/health`: the sidecar's status (`ok`, `degraded`, `starting` or `stopping`), every problem behind
  it and its capabilities. It needs no credential.

## Not in the subset

The body's side of the API stays internal: the dispatch and outbox the adapter pulls, the outcomes, observations
and answers it reports, the guard it asks before a tool runs, the Instinct routes, configuration, and every route
that changes the mind (its level, grants, the off switch). An approval comes from the owner's own chat, never from
an API caller ([asks](MIND.md#asks)).

### Current work (adapter route)

`GET` `/v1/host/executions?contact_id=<person>` returns `ColonyMindExecutionViewV1`. The owner view reads
`native_cron`, `native_kanban`, `reported_worker` when available, and `native_processes` from selected work
records. `native_processes` contains running Hermes processes and failures from the last hour, with program
names, state and timing; arguments and output are excluded. It uses `watch.hermes_home` and a 150 ms read
budget. `work_sources` reports read coverage, including unavailable sources. The top-level `items` and
`recent` lists are empty, `total` is zero and `complete` is false: this is not a complete inventory of work.
A non-owner receives the empty envelope with no reader groups or processes.

`limit` defaults to 20 (1–100). Optional `projection=request` produces the compact owner-only turn-start
work view, capped at eight items; a non-owner request for it returns 403. See [Work](WORK.md).

### Operator status and findings

`GET` `/v1/mind/status`: the mind's current status. Its owner view includes `health_counters`, a map of
counts since the sidecar process started: `loop_stalls`, `context_reads_failed`, `context_writes_dropped`,
`writes_dropped.<site>`, `mind_incomplete.<task>` and `context_card_refresh_failed`. Names appear when
bumped, so an absent name has no recorded events. Counts reset on process restart. Non-owner views omit
this map.

`GET` `/v1/host/watch/findings`: owner-only JSON list of open [job-watch findings](JOB-WATCH.md), read from
`jobwatch.db`. This operator route is outside the frozen public subset. Each finding has
`id`, `check`, `subject`, `first_seen`, `last_seen`, `state` and structured
`evidence`; times are Unix seconds. Before watch has run the list is empty. A non-owner credential gets
403; an unreadable store gets 503 `watch_findings_unavailable`. This route does not run a check or send a
notice. With the sidecar unavailable, use `colonymind watch --list` locally; that command also reports the
cumulative `delivery_failures` count.
