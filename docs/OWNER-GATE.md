# The owner gate

One gate that every unprompted message to the owner passes before it may interrupt them
(`P/mind/gate.py`, `mind.faculties.owner_gate`, on by default). It sits where the mind's messages meet the
body: `GET /v1/mind/outbox` (`Mind.outbox_ready`) asks it about every ready message to the owner, whatever
formed it (owner outreach, a notice, a task's outcome, a message another subsystem requested). Outside
scripts ask it too: an ops alert posts to `POST /v1/mind/owner-gate` before it texts the owner, and is
answered like any message of the mind's own. The gate only holds back what the existing path would send:
it never sends, grants or widens anything.

## The review

The stages run in order; the first that settles the message decides, and the verdict records the stage.

| Stage | What it reads | Verdict |
|---|---|---|
| prompted | the message's type: a reminder or heads-up on the owner's own item (`commitment_*`), an answer they requested (`outreach_answer`), the digest, a question only they answer (`link_proposal`, `cadence_confirm`, `question_again`), and what an ask sends, which waits on their yes, so holding it would hold the agent back: an ask's own message, its reminder or a list of asks (`ask_notice`), an ask's full text (`ask_in_full`), an ask shown again (`ask_show`), the plain question about an unclear answer (`ask_clarify`), the line saying what an answer did (`ask_ack`) and the line telling the owner an ask was answered in an owner channel's app (`ask_answered`), which is how an answer they did not give gets seen | `send`, recorded so dedup and spacing see it; an ask's message first waits for the pause (below). An allowlist: every other type is gated |
| dedup | the gate's own record of what went to the owner, of every kind | the same words (whitespace and case aside) on the same topic, no more urgent than when they went, inside `repeat_window_hours` (24): `drop` (the same words on another topic, or at a higher urgency, are not a repeat); a message on a topic the owner was sent inside `topic_window_hours` (12): `digest`, unless it is more urgent than everything sent on that topic there (an outcome that failed after an offer on the same item goes). A `critical` message is never settled by topic alone: it goes on to the decision with the second opinion always asked (the earlier send is in the house facts), and with no answer it goes |
| reachability | when the owner last spoke (the ledger's newest owner turn), the last send | the owner spoke inside `conversation_minutes` (10) or was sent a message inside `spacing_minutes` (10): `wait` until then (a `critical` alert does not wait), never longer in all than `max_wait_minutes` (60) from the message's first wait: past that it goes on to the decision |
| fast decision | the message and the house facts | Instinct's `owner_send` point (docs/DECISIONS.md), one decision in place of the old stages 4 and 5. Its rules tier is the deterministic prior (below): its choice among `send_now`, `digest` and `drop` settles at or above `confidence` (0.75), except a night-time `send_now` or a `critical` message on a topic already sent, which are high stakes. Where it does not settle, a decision model's fast tier is asked (in shadow until promoted on evidence; a live one may only hold for the digest or drop, at 0.80 and 0.95 at least, never send) |
| second opinion | the same, as a prompt to the main model (`task: owner_gate_review`, the `reasoning` role) | the point's deliberate tier, asked where neither the prior nor a live fast tier settled; its typed answer decides. A night-time send needs its yes: with no answer only a `critical` alert goes, and its verdict says it went unconfirmed |

**The pause.** An ask's own message, its reminder and a list (`ask_notice`) wait while the last message in the
owner's chat is the agent's own, younger than `mind.asks.quiet_minutes` (10) and not answered by the owner yet
(`ask_show` and `ask_clarify` would too, but each is queued in answer to a message and goes at once): the verdict is
`wait` (stage `pause`) until that message is `quiet_minutes` old, and the pause ends as soon as the owner writes. It
keeps an ask from landing between the agent's question and the owner's answer, where a bare "yes" would approve the
ask. What the agent last said is the later of the mind's record of the owner's chats and the last owner message the
gate let go, another ask included, so two new asks never arrive back to back into one "yes". A floor ask (urgency
`high`) waits at most `quiet_minutes` in all, any other ask at most the longer of `quiet_minutes` and
`max_wait_minutes` (60 by default), each counted from its first wait: a pause set longer than the gate's longest
wait holds an ordinary ask for the whole pause. A `send` the body has not claimed yet is checked again at the next
pull. A floor ask for work started by the owner's request goes after the reply to that request without the
conversational pause on every platform, including WhatsApp, SMS, RCS and API. Its stored note has origin `hand`,
and its `held_from` names the hand. The controller writes that hand's task record. The task must have origin
`owner_turn`, with a recorded `origin_ref` identifying the turn by platform, chat or session, and message id. The
recorded turn's sender must be the owner contact. A note or notice claiming an origin in its words proves nothing.
Once verified, the request link stays with the ask in `mind.db`, so controller cleanup and restart cannot remove
the exception. Each evaluation still checks the captured owner source and its invalidation state. The audit
records whether that evaluation verified an owner origin. A parent completion review cannot start another pause.
`delivery_ref` only selects the RCS owner notice delivery path; WhatsApp and SMS notes do not need it for this
exception. It does not bypass another ask or message the gate let go: one still younger than the pause and
unanswered holds the new ask as before. Background asks keep the full pause and bounds, as do hands started by
the mind or by a guest's turn. The body holds an `ask_notice` while a turn in its target chat has no reply, for at
most 10 minutes, and for 5 seconds after the reply, so the ask lands last and apart. A hand notice instead finishes
its wait when its native parent review ends plus the 5 second gap, or at its fixed cap. It starts no further chat
wait after either boundary. Who spoke last and when
decides, never the words. What answers a message (a show or a plain question with `answer_turn` from the owner's
words or the operator's `show`, the line saying what an answer did) goes without the pause: it is what they are
waiting for.
`mind.asks.quiet_minutes: 0` turns the pause off. With the gate off (`faculties.owner_gate: false`) nothing is
paused.

The **house facts** are what every judgment reads, so a judge is never wrong for want of context: the local
time and whether it is inside the owner's quiet hours, when the owner last spoke, how many messages went to
the owner in the last day against the typical number a day here (the last seven days), and the last few
topics sent.

The **prior** keeps the existing path's choice: a message the mind formed was approved by authority, so
`send_now` at 0.9. An outside alert gets it from its declared urgency: by day `critical` 0.95, `high` 0.9 and
`normal` 0.7 send now (`normal` sits under the confidence floor, so a configured second opinion is asked),
`low` goes to the digest; at night `critical` is `send_now` at 0.6 (always re-confirmed) and everything else
goes to the digest. A decision model that is configured but gives no typed answer (it abstained, timed out,
failed, answered off-contract or its server is down) changes nothing: the prior and the second opinion decide
as they would without it, so an outage never turns a review into a model call.

## Where each verdict goes

- `send`: the message is offered to the body at this pull and at every pull until the body claims it;
  only an ask subject to the pause is checked again as described above.
- `wait`: not offered until `until`; reviewed afresh then. A mind message the gate kept waiting that expires
  before it could go is held for the digest (stage `expiry`, settled `expired` "held for the digest: ...")
  rather than lost, whether or not the gate is still on. While a review of the message awaits a model, the
  expiry sweep leaves it to that review's verdict, and a message settled elsewhere meanwhile (a reply, an
  invalidation) stays as it was settled.
- `digest`: the mind's message is settled `expired` ("held for the digest: ..."), never the owner's verdict on
  its type. A held finding or offer lands in the digest as `Found: ...` or `Offer: ...`, as an outreach that
  expired unsent does; a held question of an earlier build is never listed; everything else, outside alerts
  included, as `Held for this digest: ...`, once. One digest lists at most 12 (oldest first) and says how many
  more wait; the rest stay held for the next digest. A listed item counts as listed only once the body reports
  that digest delivered (`digest_ref` names the digest it rides in until then); a digest that expires, fails or
  ends uncertain hands its held items back to the next one.
- A hold for the digest exists only as its record: when the audit table refuses the write, the verdict is
  `send` (stage `audit`, `scores.unrecorded`, the held reason kept in `scores`) rather than a `digest` nothing
  would deliver, and a wait that expires with its hold refused is left unsettled and offered again. Each write
  of the gate's commits or leaves nothing: a verdict whose COMMIT fails is rolled back, so a retry never reads
  an uncommitted hold as standing. Another writer's pending work on a shared connection is committed before
  the gate writes and is never rolled back by it.
- One message is listed once, as last decided: a later send, drop or hold of the same message (its source and
  reference) replaces its earlier hold in the digest.
- `drop`: the mind's message is settled `cancelled` ("dropped by the owner gate: ..."); a dropped finding is a
  repeat and is not listed.

Every review writes a `gated` transition on the row (`GET /v1/mind/why/{id}` shows its decision, stage,
reason and scores) and one row in the gate's audit table.

## Audit

The `owner_gate` table in the mind's workspace database (`mind.db`) keeps every verdict for 90 days: when, the
message (type, kind, topic, urgency, source, the text clipped to 500 characters and redacted, its evidence;
the mind's own intention id as `ref`; the message's identity, a digest of its source and reference, as `ident`), the house facts it read, the stage, the decision and its reason, `until`, and the scores (the fast
decision's probabilities and their source: `prior`, or `fast` when a live fast tier settled; the confidence;
whether it was night; the second opinion's choice and reason, or "unavailable"; and `instinct`, the id of the
decision's row in Instinct's log). Every free-text
field is redacted where it is written (the text, the topic, the evidence, the reason, the second opinion's
quoted reason, the house facts, the source), the topic's slug is kept only as a digest, an outside caller's
`id` is kept only as the `ident` digest (never its words: an id may carry a secret no redactor recognises), and
the verdict the caller gets is the redacted one. Dedup, spacing and the digest read the same table, so what the audit shows is what the gate
acted on.

`GET /v1/mind/owner-gate?limit=50&since_hours=24` returns `{enabled, settings, counts: {decision, stage},
decisions}`, newest first.

## Outside alerts

```sh
curl -s -H "Authorization: Bearer $COLONYMIND_API_KEY" -H 'content-type: application/json' \
  -d '{"topic": "nas disk", "text": "Disk 3 reports read errors.", "urgency": "high", "source": "ops"}' \
  http://127.0.0.1:7777/v1/mind/owner-gate
```

The answer is `{decision: send | wait | digest | drop, reason, stage, until, scores, facts, id, gate: on}`. The
script sends only on `send`; on `wait` it asks again after `until`. A `send` counts for dedup and spacing as if
the gate had sent it; a `digest` is listed in the next daily digest (with `mind.digest_hour` set). With the flag
off the answer is `{decision: send, gate: off}` and nothing is recorded, so a script that asks behaves as it did
before the gate existed. Turning the flag off stops new gating only: what the gate already held for the digest
is still listed in the next digests. `urgency` is one of `low`, `normal`, `high`, `critical`; `id` is the caller's own
alert id (optional) and the alert's identity: asked again under the same `id` inside `repeat_window_hours`
(the answer was lost, the send failed), the standing verdict comes back unchanged with `replayed: true` and
the same verdict `id`, and nothing new is recorded. Asked again more urgent than when it was kept back (a
`wait`, `digest` or `drop` escalated), it is reviewed afresh; a standing `send` is replayed whatever the
urgency. A condition that fires again is a new alert with a new `id` (the same words on the same topic are then
a repeat). Without an `id` a retry cannot be told from a repeat.

## Configuration

```yaml
mind:
  faculties:
    owner_gate: true
  owner_gate:
    topic_window_hours: 12
    repeat_window_hours: 24
    conversation_minutes: 10
    spacing_minutes: 10
    max_wait_minutes: 60
    confidence: 0.75
    second_opinion: true
    judge_timeout_s: 30
  asks:
    quiet_minutes: 10             # the pause before an ask enters a conversation (0: none; docs/MIND.md, Asks)
```

`colonymind.yaml` refuses unknown keys and numbers outside their ranges under `owner_gate`; a `mind.asks` value
outside its range (0 to 240 minutes) keeps its default. The second opinion is one tool-less
call on the mind's own router, inside the day's token budget (`Authority.tokens_allowed`); with no router,
no budget or no usable answer it is "unavailable" and the rules above apply. The decision model is Instinct's
fast provider (`instinct.providers`, docs/DECISIONS.md) asked the `owner_send` choice over
`send_now`/`digest`/`drop`, whose state is the house facts and the message; it ships in shadow, so until it is
measured and promoted there the prior and the second opinion decide.

## Measurement

Two no-model measurements, both in the sidecar suite:

- **mind-outreach-1**, the dev split walked through the plugin arm's code (`test_outreach_family_walk.py`,
  arm `full-plus-owner_gate` against `full`): every scenario's oracle still passes, the warranted and
  direction scenarios send exactly what `full` sends (12 warranted and direction sends, none lost) and no
  control sends more (10 and 10). `full` already sends the oracle's minimum in every control of this family,
  so on it the gate can show no loss, not a gain.
- **owner-gate-1** (`benchmarks/owner_gate/measure.py`, reference results beside it): twelve scenarios over the
  served mind, each run with the flag off (today's path, and a script that texts every alert) and on. Warranted
  (a health notice, a task outcome, a critical alert by day, a critical alert at night the second opinion
  confirms, a notice while the owner is mid-conversation, an escalation on a topic already raised): 7 sends
  off, 7 on, none lost. Controls (two kinds on one topic, an alert storm, a normal alert at night, a low alert by
  day, the same words from two sources, a critical alert at night the second opinion declines): 12 sends off, 3
  on, each at its right count.

The flag ships on (on unless shown harmful). Its measure is `all-on` against `all-on-owner_gate` (the
outreach family's `full-plus-owner_gate` against `full` measured it while it shipped off), and
`colonymind report faculties` shows what it held back.

## Limits

- Topics are compared by slug, or by each naming the other (`outreach.similar` both ways): two phrasings of one
  matter that share too few words are two topics. A mind message's topic is its declared topic, else its title
  after one of the mind's own status labels (`done:`, `Due soon:`, `Health:`, ...; `gate.TITLE_LABELS`), else
  the whole title: any other text before a colon names the subject ("Alice: approved" is not "Bob: approved").
- A `send` recorded for an outside alert is taken as sent; the gate never learns whether the script sent it.
  A script that lost the answer, or failed to send, asks again under the same `id` and gets the send again.
- The night is the configured quiet hours; the mind's own messages never reach the gate during them (the outbox
  holds everything but the digest), so its night rules apply to outside alerts.
