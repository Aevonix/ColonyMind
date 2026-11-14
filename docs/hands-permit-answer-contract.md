# Hands permit asks: the explicit answer contract

The routes are internal `/v1/mind` routes (not in the public subset of [API.md](API.md)); they keep their
shape within the 1.x line unless this page says otherwise.

A hands permit ask is the `floor_effect` note a hand's hold forms when it reaches a call that needs the owner's
yes ([HANDS.md](HANDS.md)). Its RCS copy is answered by the owner's own message (`POST /v1/mind/answers`: a
reaction, a quoted reply or the next message, unchanged). These routes let a second owner channel, platform
`companion`, read the same asks in full and answer one by its immutable id, through the channel relay's own
credential. Both channels settle the same ask row through the same settle path, and the first commit wins.

## Before any of this works: configuration

Nothing here grants anything by itself. Two settings must be live first (a separate, owner-approved step):

1. **The relay's own credential.** The companion hub (`companion-hubd`) gets a credential of its own, which only
   it holds:

   ```yaml
   credentials:
     companion-hub:
       sha256: <digest of its secret>
       owner: [companion]
       access: [answer]
   ```

   `colonymind credentials add companion-hub --owner-platform companion --access answer --key-file <path>` writes
   it and puts the secret, once, in a mode-600 file for `companion-hubd`. Credentials are read once at the
   sidecar's start, so it takes effect at the next start. A `companion` owner grant on the instance key
   (`api-key`), if one is added, serves the attribution of Chat turns through the Hermes adapter only. It never
   reaches these routes.
2. **The owner's identity.** Send either `contact_id`, the configured owner contact id (`owner.contact_id`), or
   `sender_id`, the owner's handle on the companion app (for example `owner`). `sender_id` resolves only through
   the contact store, so it needs the owner's contact record to carry the handle `companion:owner`. The
   `owner.handles` list in `identity.yaml` is read by the Hermes plugin and by `colonymind init` when it first
   creates the owner contact. Adding it to an existing instance does not add the handle to the contact store.

Without (1) every request is refused `403 answer_credential_required`. Without (2) an answer is refused `403
not_owner`. Either way nothing is settled or recorded.

## The control: the server-side credential check

Only an `answer` credential reaches these routes. The sidecar's middleware checks it before any handler runs. The
route checks it again, and so does `Mind.answer_explicit` for any caller. An `answer` credential:

- is an issued credential, never the instance key. Neither `api.key` nor any of the instance key's rotation
  secrets counts, whatever `api-key` is granted. So nothing that reads `api.key` (the Hermes gateway, an agent's
  shell or code tools, a hand) can approve a held call this way.
- has `answer` as its only access. `answer` is never in a default grant, a 1.9.0 principal's carried scopes, or
  the instance key's grant.
- names `companion` in `owner` literally. `owner: any` (`*`) is refused when the configuration loads, and never
  counts at the check.
- writes as no one (`contacts` is left out or `own`).

`colonymind.yaml` refuses any other shape when it loads. The credential reaches no other route: `/v1/mind/answers`,
`/v1/mind/asks/*`, the turn and context routes and the rest of the owner's side get `403 route_not_granted`, and
the MCP endpoint refuses it (`mcp_not_granted`).

The two answer readers (`POST /v1/mind/answers` and `POST /v1/mind/asks/{id}/shown`) take platform `companion`
from the instance key only, which is the Hermes adapter's own chat lane there. Any other credential granted the
owner on `companion` gets `403 platform_not_allowed`. Without this, such a credential could register its own copy
of any ask (a task ask, a standing grant, a people change) as shown and quote it, approving it without this
contract's checks.

Best effort only, not the control:

- `403 hand_session_refused`, for a request that names a registered hand session (`X-ColonyMind-Session`, or a
  session id in the body), catches only a caller that sends the header.
- The Hermes plugin's answer tripwire reports an agent tool call that names this route (`answer_forgery_attempt`),
  and refuses it outside the owner's own lane. It matches how a call spells the route. A raw shell can spell it
  in ways no rule sees, so this layer catches only cooperative callers.

Residual risk: anything that can read the hub's secret can answer as the owner. So can anything that can edit
`colonymind.yaml` and restart the sidecar, because it can mint itself such a credential. The line each answer sends
to RCS (below) is how an answer the owner did not give gets seen.

## `GET /v1/mind/permit-asks`

Query: `state` = `open` (default: asked and not past expiry), `settled` (everything else) or `all`; `limit`
1 to 200 (default 50). Newest first.

Response `200`: `{"asks": [<view>], "state": "open"}`. `422 invalid_state` for another `state`. `403
answer_credential_required` for any credential but an `answer` credential.

## `GET /v1/mind/permit-asks/{ask_id}`

One ask by its id, in any state. Response `200`: `<view>`. `404 unknown_ask` when the id names no hands permit
ask. Ask codes are not ids, and an ordinary ask is not a hands permit ask. `403 answer_credential_required` as
above.

### The view

```json
{
  "id": "62267c5c-3cc9-4e6b-99c1-a37b4a95468b",
  "kind": "hands_permit",
  "code": "53C",
  "hand_id": "h-000000000001",
  "status": "asked",
  "open": true,
  "outcome": null,
  "created_at": "2026-10-12T12:00:00+00:00",
  "expires_at": "2026-10-15T12:00:00+00:00",
  "terms": {
    "text": "Let this call run once. The call: irreversible deletion: terminal; {\"args\":{\"command\":\"rm /work/report.txt\"},\"cwd\":\"/work\",\"targets\":[...],\"tool\":\"terminal\"}.",
    "reply": "What is deleted cannot be recovered. Reply yes or no.",
    "reason": "held at execution from 'hand h-000000000001': ...; the exact call: ...",
    "effect_class": "irreversible_deletion",
    "call": {"tool": "terminal", "arguments": {"command": "rm /work/report.txt"}, "workdir": "/work",
             "targets": [{"kind": "file", "target": "/work/report.txt", "value": "<sha256>"}]},
    "call_digest": "<sha256>",
    "full_text_required": false,
    "full_text_delivered": true
  },
  "choices": [
    {"value": "yes", "label": "Yes", "means": "the hand may make exactly this call, once"},
    {"value": "no", "label": "No", "means": "the hand does not make this call"}
  ],
  "settlement": null,
  "winner": null
}
```

- `id` is immutable and is what the answer route takes. `code` is display only.
- `terms.text` holds the exact words a yes approves, never clipped: the RCS copy's words, or for an ask too long
  for one message, the words of its full-text message. `terms.reply` is its reply line. `terms.reason` is the
  unclipped reason (the audit routes clip it to 200 characters). `terms.call` is the exact binding the permit is
  tied to.
- `full_text_delivered` is read the way a yes is checked. It counts a full-text message the body reported sent,
  even if the hook that marks it seen never ran. Reading the view changes nothing.
- `choices` are the only accepted values. A hand's permit ask never offers `always`.
- A capture ask may additionally carry `payload`: `{kind, origin, site_hint, ttl_s,
  save_toggle_default, fields}`. Kinds are `vault_save_login`, `vault_code` and
  `vault_login_capture`. `origin` is an HTTPS origin (at most 2048 UTF-8 bytes, no
  credentials, path, query or fragment); `site_hint` is at most 256 UTF-8 bytes.
  `ttl_s` is an integer from 1 through 120. `fields` is exactly `["username",
  "password"]` for logins or `["code"]` for codes; codes have a false save default.
  Fields describe inputs and contain no values. Old asks omit `payload` entirely.
  Answers remain `yes`/`no`. The typed value travels directly to the gateway's
  `vault.capture.submit`, never to this API or its answer ledger.
- `status` is the row's current status: `asked`, `approved`, `cancelled` (a no, a withdrawal or an undone yes),
  `expired`, and others the mind uses. `open` is `false` once it is not `asked`, or once it is past `expires_at`
  before the expiry sweep closed it. After a yes, `expires_at` becomes the approval's task window.
- `settlement` (null while open) says who settled the ask since it last opened. `by` is one of `owner`,
  `operator` (a console no), `lapse` or `mind` (withdrawn, for example replaced by a newer identical hold).
  It also gives the `outcome` (`approved`, `denied`, `expired`, `cancelled`), the time `at`, and `winner`.
- `winner` is set only when the owner's own answer settled it: `{channel, choice, at, path}`. `channel` is the
  platform the answer came on (`rcs`, `whatsapp`, `sms` or another messaging platform, or `companion`). `choice`
  is `yes` or `no`. `path` is `reaction`, `reply`, `follow_up`, `list` or `explicit`. Settlements recorded before
  this release carry no channel. For those, the channel is read from the answer's message key
  (`rcs:<chat>:<message>`), and it is `null` where the key names none.

## `POST /v1/mind/permit-asks/{ask_id}/answer`

Headers: `Authorization: Bearer <the relay's secret>` and `Idempotency-Key` (required, 1 to 200 characters, not
blank). Body (unknown fields are refused):

```json
{"choice": "yes", "platform": "companion", "contact_id": "<owner contact id>"}
```

or `{"choice": "no", "platform": "companion", "sender_id": "owner"}`. Values are taken exactly as spelled, never
trimmed or lowercased. `choice` must equal one of the ask's `choices[].value` (`" yes"` and `"YES"` are refused).
`platform` must be `companion` (`"Companion"` is refused). One of `contact_id` / `sender_id` is required, and
`sender_id` wins when both are sent, as on `/answers`.

### Results

| HTTP | `code` / `result` | Meaning | Recorded under the key |
| --- | --- | --- | --- |
| 200 | `result: settled` | This request settled the ask. | yes, atomically with the settlement |
| 409 | `already_answered` | The owner's answer on some channel settled it first. `winner` says which. | yes |
| 409 | `ask_closed` | It closed without an owner answer: it lapsed (including on this request, past expiry), was withdrawn, or the operator said no. `settlement` says how. | yes |
| 409 | `idempotency_conflict` | This key was recorded with a different request (another choice, another ask, another body). | no |
| 409 | `full_text_not_delivered` | A yes to an ask too long for one message counts only once its full text was delivered to the owner in its own message. A no is accepted. | no |
| 409 | `no_owner` | No owner is configured. | no |
| 403 | `answer_credential_required` | The credential is not an `answer` credential (the instance key, any other credential, development mode). | no |
| 403 | `platform_not_allowed` | `platform` is a platform name, but not `companion`. Messaging channels answer through `/v1/mind/answers`. | no |
| 403 | `not_owner` | Who answers is not the owner. | no |
| 403 | `hand_session_refused` | The request names a registered hand session (best effort, above). | no |
| 404 | `unknown_ask` | The id names no hands permit ask. | no |
| 422 | `invalid_choice` | `choice` is not exactly an offered value. `detail.choices` lists them. | no |
| 422 | `invalid_platform` | `platform` is not spelled as a platform name (case, spaces). | no |
| 422 | (validation) | Missing header or field, unknown field, blank key (`invalid_idempotency_key`). | no |

`200` body:

```json
{"result": "settled", "ask_id": "<id>", "status": "approved", "outcome": null,
 "winner": {"channel": "companion", "choice": "yes", "at": "2026-10-12T12:00:00+00:00", "path": "explicit"},
 "settlement": {"by": "owner", "outcome": "approved", "at": "2026-10-12T12:00:00+00:00", "winner": {...}},
 "replayed": false}
```

A no settles `status: "cancelled", outcome: "denied"`, with `winner.choice` set to `"no"`.

`409` bodies have the sidecar's refusal shape, `{"detail": {"code", "message", ...}}`. For `already_answered`
and `ask_closed`, `detail` also carries the result fields above (`result`, `ask_id`, `status`, `outcome`,
`winner`, `settlement`, `replayed`).

### Idempotency

The body creates a schema-only capture with `POST /v1/mind/vault-captures`, body
`{authorization_id: <random UUID>, payload: <schema>}`. It uses the same `Mind._hold`
path as Hands, with a note that cannot launch a mind worker and an expiry matching
the payload TTL. Reply: `{ask_id}`. The creating owner/`manage` principal can poll
`GET /v1/mind/vault-captures/{ask_id}` for `{open, winner}`. These routes accept no
typed values and grant no answer capability. The relay's `answer` credential
continues to read and answer through the permit routes exclusively. The gateway
owns the authorization/session/page binding in process memory and refuses values
after cancellation, expiry, restart or use.

- A key is scoped to the presenting credential (`companion-hub`). The request digest is SHA-256 over the path's
  ask id and the validated body.
- The same key and the same request return the recorded status and body again, with `"replayed": true`. This
  holds across restarts, because the record is in `initiatives.db` next to the ask.
- The same key with any other request is `409 idempotency_conflict`. Nothing is settled.
- Only final outcomes are recorded (`settled`, `already_answered`, `ask_closed`). A refusal that changed nothing
  (403, 404, 422, `full_text_not_delivered`) is not, so the same key may be retried once the cause is fixed.
- A settling answer's record commits in the same SQLite transaction as the ask's settlement. A crash leaves both
  or neither.

### The owner is told on RCS

Every settling app answer queues one line to the owner (notice type `ask_answered`):

> Answered in the app: \<the ask's words, at most 300 characters\> -> yes. An answer given in the app is not undone
> from this chat.

A no reads `... -> no.` It goes through the mind's ordinary owner-notice outbox, sent by the Hermes body. Where it
goes:

- **To the ask's RCS anchor, when the hand's task came from the owner's RCS message.** That is the RCS chat the
  task came from, as a reply there, the same anchor the ask's own RCS card uses. Any other target is refused.
- **Otherwise, to the owner's primary handle:** the first owner handle in `identity.yaml`.

It never goes to the chat the owner last wrote in, which may be the companion chat. The owner gate always lets it
through and never drops it. Like every message but the digest, it waits out quiet hours. A replay, a late answer
or a read sends nothing more. If the line cannot be queued, the answer still stands, and the failure is logged.

### Undo

App answers have no undo. In RCS:

- **A reaction.** Changing or removing the reaction on an RCS answer's own message still undoes that RCS answer.
- **Undo words.** "undo", "cancel that" and the like name no ask. If an app yes is newer than the latest RCS
  approval in that chat and falls inside the undo window (`answers.UNDO_MINUTES`), the words undo nothing: they
  would take back the older RCS approval while the owner means the app one. The owner is told that the app answer
  stands. If the RCS approval is the newer one, undo words take it back as before.

### Winner semantics: one row, first commit wins

- The RCS reader (`Mind.read_answer`) and this route (`Mind.answer_explicit`) take the same lock and settle
  through the same `Mind._settle_ask` on the same ask row in `initiatives.db`. In one sidecar process, a second
  answer always sees the first one's settlement.
- The settling write of both is a compare-and-set in SQLite. The row moves only while it is still `asked`, and the
  status change, its history row and the idempotency record are one transaction. When two of these answers race,
  the loser writes nothing and reads the winner, even from another process.
- These are not the only writers of an ask row. The ones without the compare-and-set:
  - the console's no (`Mind.answer`, `POST /asks/{code}/no`);
  - the hands controller withdrawing an ask that a newer identical hold replaces (`Approvals._supersede`). It does
    this on purpose, even to an approved ask, to revoke its permit;
  - an RCS undo or reopen (`_undo_ask`, `_reopen_ask`), which acts on settled asks on purpose;
  - the settle paths of a standing-grant confirmation, an allow proposal and an owner question (`_activate_grant`,
    `_settle_proposal`, `_settle_question`). A hands permit ask never takes these paths.

  This is safe with one sidecar process, the only writer of `initiatives.db`. The CLI goes through its API. All of
  these writers run in the sidecar's one event loop:
  - The console's no and the hands withdrawal have no `await` between reading the ask row and writing it, so they
    never interleave with an answer's read and guarded write.
  - The undo and reopen paths run under the answer lock. So do the grant, proposal and question settle paths when
    an RCS answer reaches them, and those paths never touch a hands permit ask anyway.

  If any of these commits first, the guarded answer loses cleanly. The lapse sweep (`_expire_asks`) is guarded as
  well, so it never lapses an ask that an answer settled after the sweep listed it. A second process writing these
  rows directly would need the same guard on those writers.
- RCS ingress is unchanged: reactions, quotes and follow-ups bind exactly as before. Each settling row now also
  records `answer.channel` (the platform), which is how a later app answer reads the winner. An RCS message that
  arrives after an app answer binds nothing, because the ask is no longer open, exactly as for any closed ask.
- The winner is the first settlement since the ask last opened. If the owner undoes an RCS yes within the undo
  window (`ask_undone`), the status becomes `cancelled` and the winner stays the original yes.
- A late answer to an ask past its expiry lapses it, exactly as a late RCS answer does (`ask_closed`,
  `settlement.by: "lapse"`).

### Granting and resuming

Unchanged and entirely inside ColonyMind. The answer is recorded on the ask (`answer.path: "explicit"`,
`answer.message: "companion:explicit:<digest of the key>"`). The hand's permit reads it as the owner's
structural answer, exactly as it reads an RCS answer. Before responding, the route refreshes the hands store's
permits as `/answers` does. The hands runtime resumes the hand from the durable grant on its next poll of held
permits (every 10 seconds by default) and consumes the permit once. The companion app does nothing further. It
cannot call `/v1/mind/answers`, `/asks/{id}/shown`, `/asks/{id}/show`, any `/v1/hands` route or
`/v1/hands/permits/check` (which consumes a permit) with its credential, and must not try with any other.
