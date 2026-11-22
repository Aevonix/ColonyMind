# Grounded Ideas

The existing mind tick owns one bounded refresh task. The singleton Hermes body
collects Google reads through its existing `ctx.call_mcp` capability. There is no
additional scheduler, connector client, or model call on catalog/detail/state
reads. Collection polls on the existing body pass at most every five minutes;
changed originals and owner Build/dismiss actions use a trailing 60-second
debounce. A trigger arriving during synthesis remains pending. Turning the mind
off or pausing Hermes prevents new collection; the mind's off/body-stale checks
prevent synthesis and the mind's own start of an offer. Expiry and execution
bookkeeping still run.

Configure explicit selected native server/account IDs in `colonymind.yaml`:

```yaml
mind:
  ideas:
    accounts:
      - server: google
        account: selected-account-id
        calendars: [primary]
    definition_id: owner-authored-review-hand-id
```

Grant the existing plugin capability in Hermes configuration:

```yaml
plugins:
  entries:
    colonymind:
      mcp_allowlist: [google]
```

Successful account-bound reads establish current read availability; errors,
reconnect requirements, truncation and account mismatches are never treated as
an empty mailbox or satisfied obligation. Credentials stay in the native Google
connector. The first slice checks two explicitly selected accounts, eight recent
messages per account (including sent mail), and one selected calendar per account.
Mail searches cover 14 days; calendars cover the following 14 days. Collection
has a 20-second approximate wall bound with three-second per-call timeouts.
Coverage is deliberately partial, never proof that a document was not received.
Large originals are withheld rather than certified after truncation.

One synthesis selects up to six offers from at most 64 KB of original input.
It uses the existing `mind_deliberate` router function with fallback disabled.
There is no daily call cap. `mind_state` key `ideas:refresh` retains last-refresh
model calls, tokens, cost, model, input watermark and failure class, plus model
calls by owner-local date. Source bodies and display/execution payloads use the
existing vault; originals have scoped references in the canonical source ledger.
The commitment ledger contributes the owner's pending/overdue obligations.

The model selects a work family, original subject/summary, exact factual excerpts
and natural first-person copy. The publisher checks excerpts against decoded
source text and resolves current source versions after inference. Supplied copy
must pass `ideas_copy.validate_copy`: bounded fields, a title of at most 60
characters, one summary sentence, no title echo, tool names, raw source/account
IDs or doubled punctuation. Numbers in model prose must occur in the verified
subject or excerpts. Ordinary honorific abbreviations such as "Dr." do not count
as sentence endings. Missing or rejected prose uses verified source-specific
fallbacks, with a fixed template when source punctuation or identifiers prevent
that copy from passing. Refresh logs contain supplied, rejected and fallback
counts only. The exact inference prompt is `ideas_copy.PROMPT`.
No payment, receipt, delivery or completion is inferred.

Only marked owner task initiatives qualify for Ideas; general work APIs retain
legacy records. The six-offer cap is an eligibility bound, never a minimum quota.
On a first page with at least three cards the first three form one `hero` section;
remaining cards form `rows` sections by category, without duplicates. Thin pages
and continuation pages contain category rows. Empty sections are dropped.
Cursors bind an offset to the ordered current card IDs; a changed catalog returns
422 `invalid_or_stale_ideas_cursor` and requires a fresh first page. Both list
projections return real continuation information. The complete escaped JSON
envelope has an 800,000-byte budget, leaving room under the 1 MiB relay cap;
size-driven pagination never silently drops remaining cards.

Contract 2 on either list route emits the page, summary, presentation and label
fields the app shows. `GET /v1/host/initiatives/{id}?contract=2` returns `{card: ...}`
with the detail and items shape the app shows: `id`, `kind`, `content` (`title`,
`summary`, `detailDescription`, `fitReason`, `buildSummary`), `items`, `sharing`,
`presentation`, `iconUrl`, `mimeType` and `secondaryLabel`, the category title that
names the rows section. Each item carries `id`, `selectable`, `isSelected`,
`content` (`title`, `summary`, `buildSummary`) and `presentation`. The default
detail response remains
`InitiativeDetailResponse`. Each item has mandatory or optional selection,
content and presentation; the wire projection omits the internal `type` field.
Bound selections survive reads.
`POST /v1/host/initiatives/{id}/execute` accepts exactly
`{mode: full|selectedItems, itemIds?}` and returns
`{cardId, executionId, status: accepted}`. The existing `/respond` action remains
available, with its additional durable `taskId` receipt.

Our four original static SVG icons are selected by work kind and category.
`GET /v1/host/initiatives/media/{handle}` uses the same authenticated owner
boundary as the catalog, a finite handle allowlist, private caching, exact byte
sizes and versioned media references. Assets ship in the Python package.
`presentation.icon` supplies alt text and a complete media variant on cards and
items. Labels are empty on both, and item badges are empty. Cards carry
`actionLabel: Build`; their badges and `display_badge` appear only for accepted,
working, done or failed builds. Items carry neither `actionLabel` nor `display_badge`.
`previewImage` is optional and omitted because these offers have no verified
preview image. Sharing stays null and `capabilities.sharing=false` pending owner approval;
there is no public share target or publishing authority. Engagement and feedback
retain the existing local route/body vocabulary for the hub to map.
`POST /v1/host/initiatives/{id}/engagement` also accepts the card context of
contract 2: `event` (`visible` or `open`), `visibleFraction` from 0 to 1, the fixed
`target`, `surface` and `lane`, an `anchor` of at most 128 characters, and the
optional `value`, `anchor_kind` and `metadata`, which are validated and not retained.

Proposed offers expire at the earliest explicit ISO deadline, calendar start or
24 hours after verification. Reads check expiry, current revisions, selected
accounts and erasure without a model call. Re-reading unchanged mail does not
extend an offer's expiry. Stable action keys use owner, account, thread/event and
work family; adding supporting evidence does not create a second action. A
dismissal suppresses that action. Immaterial rewording cannot re-offer built or
in-flight work. New thread messages invalidate offers formed from the previous
thread version. Higher-ranked new offers can supersede weaker offers without a
dismissal signal. Accepted work leaves recommendations and remains available
through state, detail and the existing Hands work history. Historical results remain privately readable after
recommendation expiry while their sources remain current and unerased. An offer
the mind started (below) leaves recommendations the same way and remains
available through state and detail.

Build freezes the revision, required evidence work, selected optional items,
exact source/account bindings, result criteria and review boundary. A guarded
initiative transition persists the execution binding before dispatch. Every
queue attempt uses `idea:<executionId>` with the identical frozen brief, including
after a lost response or process restart. A different active selection is
refused. Neither the generic model's approval report nor a synthetic owner turn
can start this queue: Build requires the instance owner credential or the
authenticated Companion owner relay. Private card reads use that same boundary.
`selectedItems` accepts either optional IDs alone or IDs from the displayed item
list. Required evidence is always included, so both `[]` and `["evidence"]`
request evidence alone. Unknown or duplicate IDs and a missing itemIds list are
refused. Full scope includes every item.

Execution instructions and deliverable text come from fixed family templates,
never the model's title or build summary. Exact quoted excerpts are JSON-escaped
inside `untrustedEvidence`, explicitly marked as untrusted data, never
instructions. Retrieved email/calendar text and structured source bindings have
the same data-only status. Retries of legacy frozen briefs without read scopes are
refused by Ideas before enqueue.

## Offers the mind starts

An offer is work for the owner alone: its recipient is the owner. On the tick
that forms work (the mind on, the body alive) each proposed offer is decided by
the mind's authority, over the worker's toolsets ([MIND.md](MIND.md), Decide).
Where the level lets owner work act, the offer starts by itself: its row
becomes an `act` task with the family's fixed step as its description and a
done-check on the field `result`, queued for the board like any task of the
mind's, inside the mind's task budgets. At any other level or verdict the offer
stays a passive card, its reason naming the verdict, and is decided again on
the next such tick; the owner's Build starts it as before.

A started offer has `buildState` from its row: `accepted` while it waits for
the board, `working` once the board holds its task, `done` or `failed` when it
settles. It has no `executionId` or `taskId`. The board task's instructions
come from the same fixed family templates, with the line the report must carry,
the same review boundary (no send, spend, renewal, cancellation or external
edit) and the exact quoted excerpts JSON-escaped as untrusted data. The task
names no connector read: it works from the quotes with the worker's own read
tools, and any other effect its run reaches is held as its own ask (MIND.md,
Holds). The instructions are written when the body pulls the task, and only
while the offer's sources are current.

The task's report is its result. A report that carries the `result:` line
settles the row done, and detail shows the report as `result.packet` while the
offer's sources remain current and unerased. A report without the line settles
the row failed (`check_not_passed`). What the task came to is also written into
the agent's memory, bound to the offer's sources, so erasing a source erases
it. A started offer the board has not taken lapses at the offer's own deadline
or when its sources stop being current, and an owner dismissal cancels it.

## Integrated Hands queue

The Hands queue includes schema migrations, TaskBrief,
strict idempotency, immutable definition snapshots, authenticated owner-start
records, leased claim/bind/release pickup and the existing native runtime/private
result consumer. Build calls the real `Controller.queue_definition_task`; there
is no missing-interface 501 seam. An explicit enabled owner-authored definition
must still be selected by `mind.ideas.definition_id`. Missing configuration returns
409 `idea_review_definition_required` before binding an execution; an invalid/disabled definition is refused by
Hands. Native tools and effect permits remain authoritative.
Ideas tasks use the existing `ops` lane with an additional immutable
`source_reads` restriction in the definition snapshot. The compiled policy still
requires the owner's exact tools and connectors. It admits only the bound Gmail
message/account or the bound calendar/account and a one-day window starting at
the event, retaining only the bound event in the result. It refuses other
accounts, arbitrary queries, wrappers, shell calls and external writes,
including creating a provider draft. Draft text stays in the private result.
Ordinary `deep` tasks keep their existing allow-list.

Build returns `status: accepted` only with a durable real task ID. The existing
mind tick reconciles actual attempts; a queue's projected `running` does not
establish that an attempt has started. A completed current attempt with its
retained result branch/evidence is required for done. Result content remains
sealed and appears in private detail only while its sources remain unerased and
current. No send, spend, renewal, cancellation or external edit is authorized.

`buildStatus` and mirrored `build_status` use the vocabulary the app shows;
`buildState` keeps the engine's own state name:

| buildState | buildStatus / build_status |
|---|---|
| accepted | in_progress |
| working | in_progress |
| done | complete |
| failed | failed |

Unbuilt cards omit all three fields. This keeps an unbuilt offer idle
without claiming accepted work. State polling includes both vocabularies and
the execution/task IDs. An execute acceptance acknowledges activation; it does
not prove completed work. `viewerState.hasBuiltIdea` records a real accepted task.

Generation, expiry, dedupe, cleanup and execution write no taste, preference,
feedback or engagement signal. Only explicit UI feedback uses the feedback route.
The offline acceptance test uses a real in-process Hands controller/database,
owner-authenticated HTTP Build, queue pickup, plan/bind, accepted result and
reconciliation; no fake queue or controller is used in that test. No native
child, live model, external endpoint or taste write is needed for validation.

Tracking goals and native cron history remain outside this Ideas slice.
