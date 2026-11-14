# People: one person, every channel

Status: M5 "People" (architecture 4.7, 7.4). Implementation in this release.

ColonyMind keeps a persistent social memory for an agent: who it talks to,
what standing each relationship has, whether it may reach out to them and
how often. This document is the durable spec for the identity and contact
layer. It was first written against an audit of a deployment where the
machinery existed but attribution failed (95%+ of non-owner traffic landed on
a `default` pseudo-contact and third parties accumulated zero history).

## Design principles

1. **Attribution before analysis.** Every downstream capability (affect,
   facts, cadence, outreach, the digest) is only as good as knowing WHO each
   utterance came from. Per-message sender resolution is the foundation.
2. **A person is one contact with many handles.** The unit of identity is
   the contact; channels contribute handles (`whatsapp`, `sms`, `email`,
   `voice`, any custom gateway, ...). Matching is deterministic where safe
   (an exact handle, an E.164 number on any gateway, a normalized email),
   and a PROPOSAL where fuzzy (the same display name in a shared group
   scope), which the owner confirms. Silent fuzzy merges are forbidden: a
   wrong merge poisons two histories at once.
3. **Meet people by default.** A sender that cannot be matched gets a shadow
   contact (tier `unknown`, `may_contact='ask'`, provenance recorded) so
   history accrues from first contact. There is no switch: shadows are the
   design. A shadow is remembered, but the social drive ignores it until the
   owner sets a cadence or a tier, so a group chat never becomes a stream of
   check-in asks.
4. **A tier is standing, never permission.** Whether the agent may reach out
   is `may_contact` (never, ask, auto), raised only by the owner and lowered
   by a contact's opt-out. Familiarity, affect and relationship estimates
   never change it.
5. **Machines are not people.** Turns of system origin (cron prompts,
   skill invocations, self-echo) never write affect, facts or interactions.
   They attribute to the reserved `system` sentinel and are excluded from
   every relationship surface.
6. **Stores accept only real contacts.** The ToM APIs validate contact ids
   against the contact store; test strings and free-text names are refused
   rather than silently minting state.

## Architecture

Turn context shows the contact's recorded interaction count, last interaction
date and stored contact tier separately. It does not translate the legacy
`relationship_score` into a closeness label or percentage: that heuristic blends
contact mood with interaction frequency and recency, which do not establish the
agent's attachment. The stored score remains available for existing consumers.
Contact affect describes the contact; agent judgments have their own sourced
representation. These observations do not grant tool authority.

### 1. Sender flows with every turn

`TurnSyncRequest` gains an optional `sender`:

```json
"sender": {
  "platform": "whatsapp",           // channel kind
  "user_id": "1234567@lid",         // per-platform sender identifier
  "display_name": "Sam Rivera",   // best-effort, for shadow naming + fuzzy proposals
  "group_id": "5551212-16999@g.us"  // set when the turn came from a group
}
```

Hosts that cannot supply it lose nothing (legacy behavior); hosts that can
(the Hermes provider passes what `pre_llm_call` already carries) get
authoritative server-side attribution regardless of client caching bugs.

### 2. ParticipantResolver (sidecar, `identity/participants.py`)

`resolve(platform, user_id, display_name, group_id, channel_id) -> Resolution(contact_id, method, created)`

Resolution ladder, first hit wins:
1. **Exact transport handle**: `contact_handles(gateway=platform, address=user_id)`
   in its stored form, a verified one first, then any other that is not a
   name guess. An explicit per-channel correction, or a second contact kept
   on the same number as a separate alias, takes precedence over
   cross-gateway phone inference.
2. **Canonical identity (C1)**: `canonical_handle(gateway, address)` in
   `contacts/store.py`. An address written as a phone number (`is_e164`:
   `^\+[1-9]\d{6,14}$` after stripping spaces, dashes, dots and parentheses,
   or a phone JID `<number>@s.whatsapp.net` / `<number>@c.us`, whose digits
   always carry the country code) is the phone identity on **any** gateway
   and matches on `phone_key` whatever gateway stored it. A bare digit string
   is a user id as often as a number: it matches exactly on its own gateway,
   where a legacy bare 10-digit row is read as NANP. Nothing names a channel:
   a phone channel nobody listed resolves like `sms`. An email matches on its
   lower-cased form; anything else matches only its exact gateway and
   address. Two contacts sharing one number is ambiguous and resolves to
   none, never a guess.
3. **Scoped display-name (PROPOSAL only)**: `display_name` uniquely matches
   one member of the same group scope. A candidate association is filed and
   the sender keeps a separate shadow identity. The candidate becomes an
   owner ask (`link_proposal`); `store.confirm_link` attaches the handle
   (folding the shadow into the person through `merge`) and
   `store.reject_link` closes it. A handle an established contact holds is
   never folded: `confirm_link` refuses (`identity_handle_held`, audited
   `link_refused`) and closes the proposal; the owner merges the two
   explicitly if they are one person.
4. **Shadow contact**: tier `unknown`, `may_contact='ask'`,
   `import_source=auto:sender`, the handle attached (`Contact.is_shadow` until
   the owner files the person with a tier). Stored handles keep the transport
   gateway they arrived on (it is needed to send). A shadow's handle is the
   sender's own choice, so an owner's reference that equals it is a guess the
   owner confirms, never an exact identification (`resolve_reference`).

The resolver also OWNS the machine gate: senderless turns on machine
channels (`cron:`, `api:` prefixes, `COLONYMIND_IDENTITY_MACHINE_CHANNELS`) or
system-origin text resolve to the `system` sentinel. A turn written under a
credential's own identity (`credential:<name>`, docs/CREDENTIALS.md) never does:
it is not a person and not the system, and it stays where the credential wrote it.

### Current source attribution in active requests

An owner correction moves phone-format handles of the same number together with
bare digits on the requested gateway, or refuses if the number would still identify
two people. The refusal names conflicting handles for individual correction before
retrying; it records no change. Non-phone handles use the exact correction.
Explicitly selected sources move without rewriting their original bytes. A successful
correction invalidates source-linked descendant
answers and their projections. Reversing the correction restores direct-source
attribution; it does not revive those old derived conclusions or grant authority.

The Hermes request middleware checks cited source ownership, version and session
scope alongside the existing erasure feed. Its POST form at
`/memory/sources/erasures` accepts up to 512 exact source references and returns
`sources_current`, under the same `turns:write` scope as GET. The existing 250 ms
request budget and bounded feed pagination remain. No new queue, database or
service is involved. Changed attribution, unavailable validation, or an oversized
reference set withholds current recalled evidence and opened-source receipts.
Current human input stays intact. Deploy the backend before the updated plugin.

This checks structured native recall and authenticated source-read receipts.
It does not identify copied paraphrases or retroactively relabel native transcript
storage. An unchanged content digest does not prove that the source still belongs
to the same person. The native social-tool fixture qualifies correction and
reversal during an active request, channel separation, and inherited-child tool
revocation without an external model or a message send.

### 3. turns/sync is the attribution chokepoint

On every synced turn:
- The credential that sent it must be allowed to write as the contact it
  names and, through a `sender`, the contact the sender already is: the owner
  only from a platform the credential is granted the owner on, anyone else
  only when it writes as other people at all (docs/CREDENTIALS.md). A turn it
  may not write is refused with 403 before anything is stored; a turn that
  names nobody is the credential's own, `credential:<name>`, kept as session
  evidence and never a person. The contact the resolver settles on is checked
  against the same grant before the turn is stored.
- If `sender` is present, the resolver decides the contact (overriding the
  client-supplied `context.contact_id`, which remains the fallback).
- The contact's own words are checked for an opt-out (section 7).
- The resolved contact gets `record_interaction` (non-owner and owner alike;
  `system` never). It counts **conversations** (C3): a turn more than 30
  minutes after the contact's previous one starts a new conversation, a
  turn inside that window belongs to the current one, and
  `last_interaction_at` only moves forward.
- The comms ledger row carries the real `channel_id` (group, DM or voice).
- Affect and facts extraction runs against the resolved contact; `system`
  turns skip it entirely.
- A turn sent without a session (`context.session_id` empty, such as a gateway
  recording a dispatched task's result) becomes its own session: `derived:` and a
  digest of the caller's principal and the envelope. A retry lands on the same
  session and source, so replays and conflicts behave as with any session. A
  checkpoint or an input-linked answer still needs the caller's session.

### 4. ToM boundary validation

The contacts' affect is written by the appraisal of a contact's own turn only
(`contacts.affect_writer`), never for the owner or the `system` sentinel;
`system` is storable in comms for ops visibility.

### 5. Voice and in-person

Voice is just another channel: the deployment's speaker-identity service
resolves a voice to a canonical contact and the gateway syncs the turn with
`sender={platform:"voice", user_id:<contact_id or enrolled-name>}`. A
`voice` handle kind links enrolled voiceprint names to contacts so kiosk
and call interactions accrue to the same person as their texts. Same for
`face` if a deployment enrolls faces. ColonyMind stays generic: it defines the
handle kinds; deployments supply the recognizers.

### 6. The owner's interfaces

One interface, three doors: the router `/v1/mind/people`
(`api/routers/people.py`), the tool `colonymind_people` and the CLI
`colonymind people`. `<who>` is a contact id, a phone number, an email,
`gateway:address` or a unique name (`store.resolve_reference`: a handle
address only one contact holds comes before any name, and an ambiguous name
resolves to nobody: the 404 lists the candidates, who they are and nothing
more; with `exact=True` names are refused, which is how an owner's grant to
message a third party is kept to someone the owner identified). The owner
reads as `auto` by identity in `who` and `inspect`, whatever an older row's
column says.

| Router | Tool | CLI | Who |
|---|---|---|---|
| `GET /?q=` | `who` | `people who [q]` | everyone (a guest sees id, name and tier of the one person the query names exactly; listing is the owner's) |
| `GET /{who}` | `inspect` | `people inspect <who>` | everyone (record, digest, handles, proposals and permission history for the owner) |
| `POST /link` | `propose_link` | `people link <who> <gateway> <address>` | everyone: a candidate the owner confirms |
| `GET /proposals` | | `people proposals` | the open candidates |
| `POST /changes` | `set_permission`, `set_cadence`, `merge` | | owner; a widening change is their one-tap ask (below) |
| `POST /{who}/permission` | | `people permit <who> never\|ask\|auto` | owner |
| `POST /{who}/cadence` | | `people cadence <who> <minutes\|off>` | owner |
| `POST /merge` | | `people merge <keep> <drop>` | owner |

A mutation needs a credential granted the owner ([CREDENTIALS.md](CREDENTIALS.md))
and the caller's `contact_id` (on `/changes`, `viewer`) to be the owner, or, on
the direct routes, no viewer at all: the operator's own call from the local CLI,
which sets a change directly. `by` only labels the audit; anything else is 403
`not_owner`. The tool refuses the three mutations
outside the owner's own interactive session before asking the sidecar.

**Widening changes wait for the owner's yes (`contacts/changes.py`).** Being in
the owner's session is not the owner's yes, and nothing said in a turn stands
for it: not their words, not a page or a tool result, not a quote or a voice
note. No route has a field for words, and nothing reads them for a change.
These changes widen who gets messaged:

- `may_contact` raised (never to ask, never to auto, ask to auto);
- a cadence raised: one set where there was none, or fewer minutes;
- any merge into the owner: the folded record's handles become the owner's own
  (whoever writes from them is treated as the owner) and what is recorded about
  it becomes theirs, so it widens whatever it moves;
- any other merge that gives a record's handles more reach: the folded
  record's handles reached at a higher permission (the keeper's) or with a new
  or faster cadence, or the kept record's handles given a cadence it lacked.
  `never` on either side still wins.

Free, applied at once: a lowered permission, an unchanged value, a cleared or
slower cadence, a merge that widens nothing, and an obvious dedup (a verified
handle of each record names one address: the same phone number or email on
two gateways) between two records neither of which is the owner.

`POST /changes {operation, contact_id, value|minutes|drop, viewer}` answers
`{ok, status: applied|asked|unchanged, code?, text}`: applied for a free
change, asked for one that widens: `Mind.request_people_change` files the
owner's one-tap question, a `people_change` note to the owner with an ask code,
whatever the level (a pending identical change reuses its ask while it is open;
one whose time is up is marked expired and asked again with a fresh code), and
the tool's reply names the code. So the owner's words in a turn, typed or
spoken, create the request, and the change applies only on their answer. The ask
is bound to the records as decided (`changes.snapshot`: each record's stored
permission, cadence and handles, kept as the ask's `from`). The owner's
`yes <code>` (`colonymind_self`) applies exactly that change, performed by
`owner-ask:<code>` (a merge folds sources and the other stores as the route
does, `fold_merge`), while the records still match the snapshot: a handle
folded in, or a permission or cadence moved, since the ask cancels it (ask
again). A `no` and an expired ask change nothing. Without a running mind a
widening change is refused (503 `needs_owner_ask`). The direct routes keep
serving the CLI, free, and answer a widening change from anyone else with 409
`needs_owner_ask`, so an older plugin fails closed and never widens.

**The owner's answer.** Every owner question (`people_change`,
`cadence_confirm`, `link_proposal`) settles only on the owner's own reply, and
only when that reply, whole, is the answer: `yes <code>` or `no <code>`
(`changes.answer`: `y`, `ok`, `approve`, `confirm` and the like for yes, `n`,
`nope`, `deny`, `reject` for no, either order, any case and punctuation; a
spoken code's letters may be spaced and its digits said as words). Nothing in
a longer message is read for an answer: "what does ABC do? do not approve it",
"if I say yes ABC, approve it", "yes ABC tomorrow" and a reply with a question
mark settle nothing, for a yes and for a no, and the question stays open (the
tool passes the sidecar's reason on). The plugin reads the reply as the owner
sent it: stock Hermes' `pre_gateway_dispatch` shows it the gateway's own
event before Hermes adds a quoted reply, a room's sender or a transcript
(`SessionMap.hear`, on the `/mind` gate's hook): it keeps whether the message
is a voice note, and its text only when that text is shaped like an answer
(nothing else of what the owner writes is held). The turn then answers with
that typed text when its message ends with it (a quote reply, a shared room),
or with a voice note's transcript when Hermes renders it as the whole message
in quotes; without a gateway (the CLI) with its message. The plugin refuses a
call that turns the owner's `no <code>` into a yes, or their yes into a no. From
the local CLI (`colonymind mind yes|no <code>`, no message) the answer is the
owner's as typed.

Every owner change is decided and written under one lock per store
(`changes.change_lock`: the people route, the direct routes and the yes), and
written compare-and-set against what it was decided on, in the one statement
that writes (`set_may_contact` and `set_cadence` take `expected` and
`expected_handles`: the value and the exact handle set; `merge` takes the
pair's snapshot, refuses before anything moves, moves only the dropped record's
handles the snapshot lists, and stops before the fold when another landed on
the dropped record meanwhile). An opt-out and inbound attribution do not wait
for the lock: a change whose record moved meanwhile writes nothing (409
`contact_changed`, or the yes cancelled), and nothing is rolled back on the
shared connection, so a STOP that lands while a change is decided or written is
never written back. The fold reads both records' `never` and cadence as they
are when it commits.

A cadence set in conversation is the same kind of change: capture keeps its
rhythm and matter, never the owner's words. The tick applies one for someone
named exactly (`drives.owner_cadence`) when it only slows the contact's
cadence or keeps it, written against the cadence it was decided on (a clear
made meanwhile wins, and the next tick decides again); a new or faster one is
the one-tap `cadence_confirm` question (the duty drive asks the same for a name
match), whose yes sets it and whose no withdraws the row. The question states
exactly the interval its yes sets. A question that expires unanswered leaves
the row open and unapplied (the owner's obligation stays, and nobody is
nagged); when the owner says the cadence again after that, capture marks the
open row `restated_at` and the tick asks again under the next round's key
(`drives.cadence_question_key`).

**Merge (C2).** `store.merge(keep, drop, *, performed_by, reattribute, sources_of)`:
- every handle of `drop` moves through `identity_links.correct` (one durable
  receipt per handle, `merge:<drop>:<handle_id>`)
- the sources `drop` holds in the ledger ride on receipts of their own
  (`identity_links.move_sources`, `merge:<drop>:sources:<n>`, at most 100
  each, every source on exactly one receipt); the host's existing
  reconciliation of `pending_identity_reconciliations` re-attributes them,
  immediately for a merge through the router and otherwise on the source
  worker's next pass
- the other stores keyed by contact re-attribute through the
  `reattribute(old_id, new_id)` hooks (the comms log, the affect store and
  the commitment store, which also readdresses every owner's message granted
  to `drop`), before the sources move: a row whose contact and source
  disagree is purged as erased, so moving the rows first keeps them
- in one write guarded on `drop` still being live (a merge of the same pair
  that finished meanwhile folds nothing twice): `last_interaction_at` is the
  later of the two, `first_seen_at` the earlier, `interaction_count` the sum,
  the keeper's cadence else the dropped one's, `may_contact` `never` if
  either was (an opt-out survives a merge), the keeper's tier, the keeper's
  digest (the dropped one's when the keeper has none; never two joined), tags
  and notes appended, group memberships and identity candidates moved,
  `drop` soft-deleted
- both records are audited (`merged_in`, `merged_into`). A merge that stopped
  half way can simply be run again.

### 7. Outbound permission: `may_contact`

`contacts.may_contact` is `never`, `ask` or `auto` (architecture 7.4). The
owner is `auto` by identity. A new shadow, an introduction, a provisioned
handle and a migrated contact start at `ask`; a contact whose old
`interaction_allowed` was false migrated to `never` (migration
`006_may_contact.sql`, which drops `interaction_allowed`; the store needs
SQLite >= 3.35 and says so at connect).

- `store.set_may_contact(contact_id, value, *, by, reason, expected,
  expected_handles)` is the owner's path, in any direction, audited
  `may_contact_set`; from the owner's session a raise is their one-tap ask
  (section 6); with `expected` (and `expected_handles`) it is compare-and-set
  against that stored value and handle set (`ContactChanged` when an opt-out,
  another change or a new handle moved it meanwhile).
- `store.lower_may_contact(contact_id, *, reason, source_ref)` goes to
  `never` only, audited `opt_out`. Two detectors end there: the phrase match
  in `contacts/optout.py` on the contact's own words (a bare `STOP`, also
  behind a gateway's bracketed timestamp or sender header, `unsubscribe`, "don't message/text/contact me", "no more messages from
  you", "stop the check-ins", "I'd rather you didn't message me", "leave me
  alone", "remove me" and close variants; never for the owner) and the
  appraisal call's `opt_out` flag. A phrase counts only as a request of its
  own, at the start of a sentence or ending its clause: "remove me from the
  Thursday thread" or "the kids won't leave me alone today" is not one.
- Nothing else writes the column: `update()` refuses it, a tier promotion
  leaves it alone, and learning never touches it.

### 8. Cadence and the social drive's candidates

`contacts.cadence_minutes` is the owner's check-in cadence (`set_cadence`,
audited `cadence_set`; null clears it; a new or faster one from the owner's
session is their one-tap ask, section 6). `compute_cadence_overdue` honours it
exactly when set and otherwise estimates the contact's rhythm from their
conversations (active span / conversations), so a chatty contact no longer
collapses to the floor. `store.social_candidates()` lists what the social
drive may consider: not the owner, `may_contact` is not `never` and the owner
set a cadence or the tier is `regular` or above. Shadow and group-only
contacts are never listed, so a group of strangers raises no check-in and no
ask (`test_people_acceptance.py` runs ten through the real sender resolution
and a real tick).

### 9. Per-contact digest

`contacts.digest` (with `digest_sources`) is a short template record of a
person: who they are, how they are reachable, when you last talked, what is
open and the top claims from their own sources. It never carries what the
owner set for them (permission, cadence, who introduced them): the digest is
shown as "About this person" when that person is the viewer and composes the
messages sent to them, so those stay in the owner's `inspect`, read from the
columns. The mind writes it daily for contacts with a conversation in the last
day (`store.set_digest`, sources `["template"]`). While a generated digest can
be written (the memory faculty's, with consolidation on and a router), the
template fills only an empty digest or its own earlier template. The section
is cut at the template's 600 characters whoever wrote the column. With the
people faculty off there is no digest section. It replaces the old
relationship briefs.

### 10. Remediation of poisoned history (deployment runbook)

- `default`: stop new person-writes (machine gate); exclude it from every
  relationship surface; keep rows for ops history. No deletion (part of it
  is genuine pre-fix owner traffic).
- Test residue (`validate-*` and other non-cid ids): purge from the affect
  and facts stores; the ToM boundary validation prevents recurrence.

## Config

| Env | Default | Meaning |
|---|---|---|
| `COLONYMIND_IDENTITY_MACHINE_CHANNELS` | `cron,api,internal` | Channel prefixes whose senderless turns are `system` |

## Test plan

- `test_people_store.py`: migration 006 on a pre-006 store; `may_contact`
  moves only through the owner path and lowers only to `never`; cadence and
  digest; conversations (C3); `social_candidates`; `resolve_reference`; the
  C1 probe (three phone senders on `sms`, `whatsapp` and a custom gateway
  are three contacts on every later turn, no orphans; one number on two
  gateways is one contact; a non-E.164 custom id stays gateway-scoped);
  merge (C2) folds, moves handles, candidates and sources exactly once,
  calls the hooks and can run again after stopping half way; link proposals.
- `test_people_router.py`: every route; the owner check (403 for anyone
  else, accepted for a credential granted the owner, from the owner's session or
  naming no viewer); a guest sees only who someone
  is; a merge re-attributes ledger sources, commitments and sourced affect;
  evals 7.2 test 4. `tests/hermes_adapter/test_people_contract.py` runs
  evals 7.2 tests 4 and 8 through the real plugin against these routes.
- `test_people_acceptance.py`: a group of ten strangers and invariant
  episode 2 on the real stores; `test_people_family_walk.py`: the
  `mind-people-1` dev split and the delegated chase walked through the
  plugin arm's code with a scripted model, graded by the family's oracles.
- `test_optout.py`: the family's phrasings, close variants and near misses.
- `test_people_changes.py`: what widens (any merge into the owner too), the
  obvious dedup, the owner's answer (their whole reply, typed or spoken; nothing
  more or less), `/changes` and the direct routes on a real store and a real
  mind (no field for words, asked, applied, reused, yes and an unsaid yes or no,
  stale handles, a handle landing while a yes is written or a fold runs, an
  opt-out racing a change, a yes or a failed compare-and-set, expired);
  `test_cadence_words.py`: a captured cadence asks whatever was said, its yes,
  no and expiry, a free slower cadence that never lands over a clear, the
  renewal after expiry; `tests/hermes_adapter/test_people_confirmation.py`:
  the plugin on stock Hermes against the real sidecar (a raise asks whatever
  the owner said, a page read in their session, a yes their reply does not say
  leaves the ask open, a spoken yes settles it) and the gateway's event (a voice
  note's yes, a spoken no never turned into a yes, a typed quote reply, the
  record speaking only for its own turn).
- `test_people_cli.py`, `tests/hermes_adapter/test_tools_commands.py` and
  `test_guard.py`: the CLI and the tool, owner-only mutations refused in a
  guest, worker and cron session, cron delivery to a `never` contact blocked.
- Resolver ladder: `test_resolve_messaging_handle.py`,
  `test_identity_corrections.py`, `test_signals_attribution.py`.

## Source erasure

Source forgetting uses `POST /v1/host/memory/curated/sources/forget` with explicit `person_id` and source ids,
or one unambiguous whole-message `old_text` in its `session_id`. It appends a content-free tombstone and durable
receipt in `curated_memory_events`. The shared fence suppresses forgotten sources, same-session message copies
and recorded dependencies in text/voice recall, curated read/search and allowance repacking. Historical source,
claim, affect and projection rows remain intact. No cleanup sweep or host-database mutation runs.

`source_erased: true` remains a compatibility key and means that the tombstone was recorded. The response also
contains `source_ids`, the curated `receipt`, `scope: curated_source_tombstone` and a note explaining retention.
Repeating the same normalized selection returns the same receipt. A retired destructive URL returns
410 `memory_forget_moved`; the tool reports that nothing was removed. Ship the sidecar with or before the adapter.

Affect event responses expose `source_lineage` and `evidence_basis`. Events
with unknown or explicit independent origins stay labeled
`unlinked_observation`. This is source-aware retention, not proof that an
inferred state is correct. Raw source retention, learned assertions and
inferred relationship state remain distinct, and the attribution and
authority rules above still govern which conversations can update these
stores.
