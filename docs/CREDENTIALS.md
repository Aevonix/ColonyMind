# Credentials: who may write as whom

Every request to the sidecar carries a credential. Each credential is bound to the contacts and channels
it may write as, and the sidecar checks every turn against that binding before anything is stored. A turn
the credential may not write is refused with a clear error; it is never quietly re-attributed.

Why: with one key shared by every client, any client can name the owner in a turn's `contact_id`. An
integration configured that way (a game bot posting its game log, say) writes every line as the owner's:
the lines become the owner's claims and appraisals, and the nightly consolidation asks the owner which of
two things "you told me" is right. Who said something is decided by where it came from, and only a few
channels speak for the owner.

## The credentials of an instance

- `api.key` is the instance key: the Hermes adapter's credential. `colonymind init` writes it and the
  adapter reads it (`plugins.colonymind.key_file`). Any client that shares the file shares its binding.
- `colonymind credentials add <name>` issues another credential, one per integration, voice gateway or
  script. Only its secret's SHA-256 digest is kept.

`colonymind.yaml` binds them:

```yaml
credentials:
  api-key:                            # api.key: the Hermes adapter
    owner: [whatsapp, sms, rcs, voice]
    contacts: all
  voice-gateway:                      # colonymind credentials add voice-gateway --owner-platform voice
    sha256: "<digest of its secret>"
    owner: [voice]
    contacts: all
  game-bot:                           # colonymind credentials add game-bot
    sha256: "<digest of its secret>"
    owner: []
    contacts: own
  relay:                              # a credential whose secret is being rotated
    secrets:
      - {sha256: "<digest of the old secret>", accept_until: "2030-01-08T00:00:00+00:00"}
      - {sha256: "<digest of the new secret>"}
    accept_until: "2030-06-01T00:00:00+00:00"
    owner: []
    contacts: own
  front-desk:                         # bound to exactly two persons, writing only
    sha256: "<digest of its secret>"
    owner: []
    contacts: [cid-front-desk, shared]
    viewer: cid-front-desk
    access: [write]
```

| Key | Meaning |
|---|---|
| `owner` | The platforms a turn may be attributed to the owner from; `any`: every platform. Empty: never the owner. The default for `api-key` is `whatsapp, sms, rcs, voice`; nothing else speaks for the owner until it is listed, the CLI (`cli`) included. Listing `cli` (or `any`) lets anything that can start a Hermes chat on the machine write as the owner, and so answer the owner's asks (The Hermes adapter and a voice gateway, below). |
| `contacts` | `all`: anyone else the sidecar resolves from a sender or the body names. `own`: only the credential's own identity, `credential:<name>` (the default for an issued credential without `owner`). A list of contact ids: exactly those persons (the shared and global persons only when listed by name; the owner only through `owner`). |
| `viewer` | With a list of contacts: the person a request that names nobody acts as. Without one, such a request is refused (`person_binding_required`). |
| `senders` | With a list of contacts: `all` lets a sender resolve to anyone (the owner, shared and global persons still only as granted), and a turn with a sender starts from its `viewer` whatever contact the body names, the sender deciding; by default a sender must be one of the listed persons. |
| `access` | What it may do: `write` (the turn routes), `read` (context and memory), `manage` (every other route, the owner's side of the API), `note` (notes through the MCP endpoint, always under its own identity; no route is a `note` route, and `write` includes it), `answer` (only an owner channel relay's own credential: it reads the hands permit asks and records the owner's explicit answer to one, `/v1/mind/permit-asks/...`, and nothing else; it stands alone in `access`, names its channel in `owner` one by one, never `any`, writes as no one, and the instance key never has it: [hands-permit-answer-contract.md](hands-permit-answer-contract.md)), `observe` (a hub's credential for outside observations: below). The default is `[read, write]`, and `manage` too for a credential granted the owner; no default grant includes `answer` or `observe`. |
| `observe_sources` | With `access: [observe]` only: the sources it posts outside observations for, a non-empty list of source identifiers. |
| `scopes` | In place of `access`, for a credential carried from a 1.9.0 principal: that principal's scopes, as they were. It reaches exactly the routes 1.9.0's rules gave them (`*` every route, `api:access` the unscoped rest), and `unscoped_api: false` keeps a principal that had `allow_unscoped_api` off from the unscoped rest. |
| `sha256` | The digest of an issued credential's secret. `api-key` has none: its secret is `api.key` (or `COLONYMIND_API_KEY`). |
| `secrets` | In place of `sha256`, while a secret rotates: every secret the credential accepts, each a digest with an optional `accept_until` of its own. For `api-key`, the secrets it accepts besides `api.key`. |
| `accept_until` | An ISO 8601 time (UTC when it names no zone). From then on none of the credential's secrets is accepted (401). `api.key` itself never expires. |

A section that is missing is read as the default above. `colonymind init`, `colonymind upgrade` and
`colonymind credentials` write it explicitly. The sidecar reads it at start: restart it after a change
(`colonymind service restart`). The sidecar log names every credential, what it may write as and until
when, never a secret. Expiry is checked on every request, so an expired secret is refused without a
restart.

## Companion owner goals

The production server treats a credential with explicit `owner: [companion]` and
`access: [manage]` as the owner's private phone relay for `/v1/mind/owner-goals`.
Its name is not authority; the authenticated server-side grant is. `owner: any`
alone does not identify a phone relay. For example:

```yaml
credentials:
  companion-hub:
    sha256: "<digest of its secret>"
    owner: [companion]
    contacts: all
    access: [manage]
```

The server binds an owner-only audience, permits requested writes and confirmation
of retained proposals, and verifies exact resolved effects and receipt evidence in
the mutation transaction. Every mutation still requires an `Idempotency-Key` UUID;
replays require the current authenticated grant. Caller headers, `source_id`, and
body fields cannot select owner authority. Agent proposal creation remains denied
until a verified owner-turn source is integrated. Other owner channels still need
a trusted audience/verifier integration. Artifact resolution and scheduled-run
destination evidence retain their separate fail-closed defaults.

## Observe credentials

A hub that posts the owner's outside observations gets an `observe` credential, the least it needs:

```yaml
credentials:
  sensor-hub:
    sha256: "<digest of its secret>"
    owner: [companion]
    access: [observe]
    observe_sources: [sensor]
```

It reaches only these routes, and only for the sources in `observe_sources`:

- `POST /v1/mind/observations/sources/{source}`: the source's observations (docs/API.md, Outside observations).
- `GET` and `PUT /v1/mind/observations/sources/{source}/cap`: the source's daily cap, read and set as the owner.
- `GET` and `PUT /v2/host/conversations/observed/{conversation_id}`: a conversation observed from a source, which
  the body names (`source`); the route refuses a source the credential does not list.

Any other source is refused with 403 `source_not_granted`, and every other route with 403 `route_not_granted`: it
writes no turns, reads no context or memory, and reaches nothing else of the owner's side of the API. `observe`
stands alone in `access`, the credential is granted the owner on platforms named one by one (never `any`), it writes
as no one (`contacts: own`), and the instance key never has it; the configuration is refused otherwise. A credential
with `manage` and the owner grant (the instance key by default) reaches the same routes for every source.

## Owner SMS identity

Configure the owner's SMS and RCS numbers independently in the deployment's local `colonymind.yaml`:

```yaml
owner:
  contact_id: "<existing owner contact id>"
  sms_handle: "+15550000011"          # fictional placeholders; replace locally
  rcs_handle: "+15550000011"
credentials:
  api-key:
    owner: [whatsapp, sms, rcs, voice]
    contacts: all
```

Alternatively, set `COLONYMIND_OWNER_SMS_HANDLE` or `COLONYMIND_OWNER_RCS_HANDLE` in both the sidecar and gateway
process environments. It overrides the file; an empty value disables that
binding. Use one E.164 string including `+` and the country code. The number is
read from local configuration on each lookup and is never written into the
contact database by this setting. Changing or removing it also invalidates the
adapter's cached owner classification for that platform. Another platform's number
alone does not establish either phone channel's owner identity.

Previously configured explicit SMS/RCS handles in `identity.yaml` or the contact
store remain separate bindings. Remove those locally too when retiring a
number; clearing the new setting does not edit existing identity records.

Keep integrations on separate credentials with `owner: []`. The phone setting
does not grant them an owner platform or permit them to impersonate the owner
by supplying the configured number. The API still checks the credential before
storing a turn or accepting an owner answer.

Each phone transport must authenticate ingress before exposing an owner sender to
the gateway. A bridge that accepts caller-supplied sender fields without
verification must retain its guest identity. The adapter's shared `sms_ingress` helper
accepts only the configured sender in a direct chat with transport-generated
verification evidence from the trusted loopback HTTP bridge. SMS accepts HMAC-SHA256
or ECDSA-P256-SHA256; RCS requires the phone device key (ECDSA-P256-SHA256). The bridge must verify the
signature, timestamp, nonce replay and body digest itself; a `verified` flag
from an incoming message body is not evidence. Other senders remain guests.

The deployment’s Pixel RCS phone plugin calls `owner_rcs_sender` only on drained bridge
envelopes and retains its exact conversation ledger check. Unverified messages
use separate guest identities and sessions. Add `rcs` to `hands.platforms` and
the instance credential owner grant after verifying detached completion delivery.
This setting grants identity; owner asks, permissions and action gates still apply.

## What a turn may be attributed to

A turn's platform is its sender's (`sender.platform`), else the part of the channel it names before the
first colon (`context.channel_id`: `cli`, `whatsapp:+15550000000`). A channel the sidecar derives itself
never counts. The check runs on `POST /v1/host/turns/sync` and on the `PUT /v2/host/turns/...` routes
that carry a turn, before the turn is reserved or stored, an ordinary turn and a checkpoint alike (a tool
observation, which has no platform, is the owner's and needs a credential granted the owner):

1. A credential bound to its own identity (`contacts: own`) writes only as `credential:<name>`: a body
   that names nobody, or names that identity, is written there; one that names anyone else is refused, and
   so is any `sender` (a sender is a person).
2. Any other credential may attribute a turn to the owner (or, unless it lists them by name, to the shared
   and global persons) only from a platform in its `owner` list. That holds for the owner named in
   `contact_id` and for a sender who already is the owner, whichever contact id the body carries.
   A credential bound to a list of contacts writes only as those persons, named in the body or reached
   through a sender, unless it takes `senders: all`.
3. A turn that names neither a contact nor a sender is the credential's own: `credential:<name>`
   (`credential:api-key` for the instance key). A turn with a sender and no contact is the sender's.
4. No credential writes as another credential's identity.
5. The contact a turn ends up attributed to, once the sidecar has resolved its sender, is checked again:
   a sender who became the owner in the meantime, or someone a list-bound credential may not write as,
   is not stored (`accepted: false`, `skipped_reason: owner_not_granted` or `contact_not_granted`).

| Status | `detail.code` | When |
|---|---|---|
| 401 | | the secret matches no credential, or its time is up |
| 403 | `owner_not_granted` | the turn would be the owner's (or shared or global) from a platform the credential is not granted, or the credential never writes as the owner |
| 403 | `contact_not_granted` | a credential bound to its own identity, or to a list of contacts, named someone else, or a credential named another credential's identity |
| 403 | `sender_not_granted` | a credential bound to its own identity sent a sender, or one bound to a list sent a sender who is none of its persons |
| 403 | `person_binding_required` | a credential bound to a list of contacts, with no `viewer`, named nobody |
| 403 | `route_not_granted` | the route's operation (write, read, manage or observe) is not in the credential's `access`; by default a credential that may never write as the owner reaches only the turn and context routes |
| 403 | `answer_credential_required` | a hands permit ask route (`/v1/mind/permit-asks...`) from anything but an `answer` credential: the instance key (by `api.key` or a rotation secret), any other credential, development mode |
| 403 | `source_not_granted` | an `observe` credential named a source it does not list in `observe_sources` |
| 403 | `platform_not_allowed` | `POST /v1/mind/answers` or `POST /v1/mind/asks/{id}/shown` naming an explicit-answer channel (`companion`) from any credential but the instance key |
| 403 | `not_owner` | a credential that reaches the rest of the API without the owner grant called a route of the mind that has no non-owner view, or the sealed store (`/v1/secrets`) |
| 403 | `approve_required` | a request that cites no owner turn tried to widen: a yes to an ask (`POST /v1/mind/asks/<code>/yes`), a standing grant that names no conversation, or a raised permission, a new or faster check-in or a merge that widens reach through a people route naming no viewer. Refused for every credential, the instance key included: the owner says yes in their own chat (docs/MIND.md, Asks: console and API answers) |
| 409 | `needs_owner_ask` | a change that widens who gets messaged, sent from the owner's session through a direct people route: it is the owner's ask instead, filed through the plugin's `POST /v1/mind/people/changes` and answered in their chat |

The message says which credential was refused and what it may write as. Nothing is stored and no turn id
is reserved, so a client may send the same turn id again with a body it is allowed to write.

## A credential's own identity

A turn written as `credential:<name>` is kept as session evidence: it can be recalled within its own
session, and nothing else reads it. It is never projected into claims, appraisals, commitments or
opinions, so it can never raise a contradiction question to the owner, and it is not a person: it records
no interaction, no communication and no opt-out. An integration that plays a game, replays a log or runs
a batch writes there. Nothing re-attributes it: not a sender, not a machine channel (`cron:`, `api:`) and not
the words a machine's turn carries ("System note: ...", a system tag), so it stays the credential's and the
owner forgets it by that identity and its source.

## The owner's side of the API

Only a credential granted the owner on at least one platform speaks for the owner. Every credential goes
through the same check, the instance key included (`colonymind credentials set api-key --no-owner` binds
it like any other):

- A credential reaches the routes of the operations in its `access`: `write` the turn routes
  (`/v1/host/turns/sync`, `/v2/host/turns/...`), `read` the context and memory routes
  (`/v1/host/context/assemble`, `/v1/host/memory/read`, `/v1/host/memory/search`,
  `/v1/host/memory/recent`, and the memory inventory and a claim's history, `GET /v1/host/memory/inventory`
  and `GET /v1/host/memory/inventory/claims/{id}/history`, which a credential bound to a person reads for that
  person only), `manage` every other route. By default a credential that may never write as
  the owner (`owner: []`) reaches only the turn and context routes. A credential reads only what it may
  write as: a read that names no person is its own identity's (its `viewer`'s when it has one), never the
  unscoped owner view, and no credential without the owner grant reads the owner's lanes.
- Everything else (the mind, people, commitments, erasure, configuration) is the `manage` operation,
  which by default only a credential granted the owner has.
- The owner's view is the credential's, never something a request asserts. Every read of the mind that
  takes a `viewer`, `contact_id` or `by` (its state, log, asks, lessons, opinions, people) shows the
  owner's view only to a credential granted the owner; what the request sends can narrow that view to a
  guest's (an empty `contact_id` is a session with no owner in it, a worker's or a cron run's), never
  widen one. Any other caller gets the non-owner view: the switches, the rows addressed to a person it
  may act as, no asks, lessons or owner-audience opinions. A credential that has `manage` without the
  owner grant reaches nothing else of the mind and nothing of the sealed store (`not_owner`), whatever
  lane, contact or session a request names.
- Outside the Companion owner-goal actions above, no credential approves anything by itself. A yes to an ask,
  a standing grant and a change that widens who gets messaged come only from the owner's own message in their
  chat (docs/MIND.md, Asks). A request that cites
  no owner turn (the key holder's CLI, a script) may narrow and never widen: a widening is refused 403
  `approve_required` whatever its credential. There is no `approve` operation: an agent with a shell could mint
  any credential the owner can. `POST /v1/mind/answers`, the route a host hands the owner's own message to, takes
  it only from a platform the credential is granted the owner on, as for a turn capture. The one exception is an
  owner channel relay's `answer` credential, issued by the owner, which records the owner's explicit answer to a
  hands permit ask by its id and reaches nothing else; the readers (`/answers`, `/asks/{id}/shown`) take that
  channel from the instance key only ([hands-permit-answer-contract.md](hands-permit-answer-contract.md)).
- The people routes' changes (who may be contacted, a cadence, a merge) and an opinion's withdraw or
  reconsider need a credential granted the owner, from the owner's session (a viewer `contact_id` that is
  the owner) or naming no viewer (the key holder's CLI). A narrowing applies at once from either. A change
  that widens who gets messaged is refused on the direct routes: from the owner's session with 409
  `needs_owner_ask` (the plugin's `POST /v1/mind/people/changes` files it as the owner's ask, which their own
  answer applies), naming no viewer with 403 `approve_required`. A standing grant is the owner's ask from their
  own conversation, or the operator's `mind.grants` in `colonymind.yaml`: a request that names no conversation
  (no `contact_id`, `session_id` or `message`) is refused 403 `approve_required`. A body's `by` only labels
  the audit.

Development mode (no key; loopback callers only) is a principal of its own, `anonymous-dev`, granted no
owner platform: its turns are never the owner's (a sender who is the owner is refused like any other
credential's), its reads never the owner's lanes, and besides the turn and context routes it reaches only
the mind's non-owner views (`route_not_granted` for the rest). Write `api.key` for the owner's side of the
API.

## An operator's test through a live channel

A check sent through a channel that speaks for the owner (a qualification prompt through the voice
gateway, a probe after a deploy) is not the owner's words. Mark it, and the sidecar keeps it under the
sending credential's own identity (`credential:<name>`), which nothing projects, appraises, captures or
questions, even on a platform the credential may write the owner from. Only these trusted signals mark a
test:

- the request header `X-ColonyMind-Test: 1`, or
- the turn's text opening, at its first character, with exactly `[operator-test]`, or
- the qualification harness's provenance line, in the exact shape the harness writes, as the text's first
  line: `[qualification-<run>-<UTC time as YYYYMMDDTHHMMSSZ>-<hex run id>; operator-authored qualification,
  not a recorded human conversation]`.

On a checkpoint, the header or any of its user messages opening with one of those marks makes the whole
checkpoint a test. Anything else is the owner's: a mention later in the text, another spelling, and a
bracket that quotes or negates a mark (`[This is not operator-authored] ...`) included. A marked turn is
checked like any other first, so a test the credential could not write unmarked is refused, not
rewritten. Tests that go through a live channel must carry one of these marks.

## The Hermes adapter and a voice gateway

The adapter names the platform of every turn: the sender and their chat on a messaging platform, the lane
itself (`cli`, `cron`, `api`) for a turn without a sender, the platform the memory provider was initialized
on when no session binds one (stock Hermes' CLI), and the participant's platform on a checkpoint. A CLI
conversation is therefore the owner's only on an instance that grants `cli`.

Granting `cli` (or `any`) has a cost: a turn from the terminal is whatever a local `hermes chat` says, whoever
started it, so anything that can start one on the machine writes as the owner, an agent's own shell included, and
can answer the owner's asks. `colonymind credentials` says so for each credential that grants it (in `list`, and
after `add` or `set`), and `colonymind doctor` warns (`cli-owner-platform`) when an owner lane runs a shell on the
same machine, printing for each such credential the `colonymind credentials set` that keeps its other platforms and
drops the terminal (docs/MIND.md, Asks: console and API answers).

When the sidecar refuses a turn (`owner_not_granted`, `contact_not_granted`, `sender_not_granted`, or
`accepted: false` with one of those as `skipped_reason`), the adapter never sends it again as another
contact. It logs the refusal and counts it as a failed capture: the general plugin keeps the refusal on
the turn's outbox row (`attribution refused: <code>`) and `/mind status` counts those turns; the memory
provider, as the turn writer, counts them in its diagnostics (`attribution_refusals`, `last_sync_error`)
and no longer reports the connection as healthy. Grant the platform and later turns from it are captured.

A voice gateway posts each spoken turn with `sender: {platform: voice, user_id: <contact id>}` and
`channel_id: voice`. With `api.key` it writes the owner's turns because `voice` is in the default grant;
with a credential of its own it needs `owner: [voice]` and `contacts: all` (the callers are contacts).

## An MCP client

An agent that speaks the Model Context Protocol reads through the endpoint `POST /v1/mcp` with a credential of its own
([MCP](MCP.md)). That credential decides who its tools act for, as it decides who its reads here are for, and its
`access` decides which tools it gets: `read` the reads, `note` (or `write`) a note, which is kept under
`credential:<name>` like any turn written there. There `read` also covers the mind's reads and the open commitments,
which the API's own routes keep under `manage` with every change. The instance key is refused at that endpoint.

## Managing credentials

```bash
colonymind credentials list                                   # every credential, what it may write as, until when
colonymind credentials add game-bot --key-file ~/game-bot.key # its own identity only; secret in a mode 600 file
colonymind credentials add voice-gateway --owner-platform voice --contacts all
colonymind credentials add coder --owner-platform mcp --access read --access note   # an MCP client: no turns
colonymind credentials add companion-hub --owner-platform companion --access answer \
    --key-file ~/companion-hub.key          # an owner channel relay: answers hands permit asks, nothing else
colonymind credentials add sensor-hub --owner-platform companion --access observe \
    --observe-source sensor --key-file ~/sensor-hub.key   # a hub: posts one source's observations, nothing else
colonymind credentials set api-key --owner-platform whatsapp --owner-platform sms \
    --owner-platform rcs --owner-platform voice --owner-platform cli
colonymind credentials remove game-bot
colonymind init --owner-platform whatsapp --owner-platform cli   # the instance key's owner platforms, in full
```

Without `--key-file` the secret is printed once. `set` replaces a credential's `owner` list (`--no-owner`
empties it) and, given `--access` (once per operation), its access, and given `--observe-source` (once per
source), an `observe` credential's sources; it keeps its secrets and expiry. The instance key cannot be removed; replace `api.key` to
rotate it. To rotate an issued credential's secret, list the old and the new digest under `secrets`, give
the old one an `accept_until`, and restart the sidecar.

`colonymind init` and `colonymind upgrade` print what each credential may write as, and name any platform
the owner has a handle on in `identity.yaml` that the instance key may not write the owner from.

## Upgrading

`colonymind upgrade` on an instance whose `colonymind.yaml` names no credentials records the grants it now
runs under (the default for `api-key`) and reports what each credential may write as; an instance that
names them is left as it is.

An instance that already runs 1.10 or later has a `colonymind.yaml`: 1.10 introduced it as the instance's
one configuration file, and 1.9.0 never wrote or read it. 1.10 accepted only the instance key, so upgrading
such an instance records only `api-key` (the owner from WhatsApp, SMS, RCS and voice, and the whole API,
as 1.10 gave it). A 1.9.0 keyring (`api-keyring.json`) still on disk from before 1.10 is not read: the
report names it as retired and not carried, and the file is left for the owner to remove.

Only a 1.9.0 instance (an `instance.json` and no `colonymind.yaml`) migrates its keyring, resolving every
principal's grants before anything is written. Every live principal but the adapter's becomes a credential
with exactly the authority it had, neither narrower nor broader:

- the same persons: its `person_ids`, its `viewer_person_id` and the owner, shared and global persons its
  `audiences` name, as a list of contacts (the owner through `owner`), with its viewer as `viewer`;
- the same senders: `turns:resolve-sender` becomes `senders: all`;
- the same routes: its scopes are kept as they were (`scopes`, and `unscoped_api: false` when
  `allow_unscoped_api` was off), and reach exactly the routes 1.9.0's rules gave them;
- the same owner authority: a principal granted the owner writes the owner from every platform
  (`owner: any`). Its declared platforms (`turn_ingress_platforms`, its attested contact platforms) only
  told 1.9.0 which senders to attest; they never limited where it wrote the owner from.

What has no counterpart (a principal's attested contact grants, the contacts its senders had resolved to,
as persons to name) is named in the report.

The principal whose secret the adapter used (`COLONYMIND_CLIENT_API_KEY`, else the first live one) becomes
the instance key: the Hermes adapter's credential, which writes and reads as anyone the sidecar resolves
and reaches the whole API, and writes the owner from every platform (the default ones when the principal
was not granted the owner). `api.key` is one of its secrets that never expires, or a
new secret when all of them retire; its other live secrets, and a live `COLONYMIND_API_KEY`, stay accepted
as the instance key, each until its own expiry. Every secret the 1.9.0 sidecar still accepted is kept,
with its own `accept_until` and its principal's.

The upgrade report lists every principal: what it could do before and what its credential may do after,
in the same words, or why it is not carried (revoked, expired, no live secret).

A deployment whose owner talks to the agent on a platform outside its grant (a CLI, another messaging
platform) grants it after the upgrade; until then the sidecar refuses those turns as the owner's and the
adapter counts each as a failed capture.
