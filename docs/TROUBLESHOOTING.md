# Troubleshooting

Each entry is a symptom, how to check which cause you have, and the fix. Start with `colonymind doctor`: it
checks the install, the Hermes keys, the adapter, the service and the running sidecar, and prints a remedy for
each failure. Hermes' own log is `hermes logs` (`--level WARNING` for warnings and errors); the sidecar writes
its log to `<instance>/service/sidecar.log` (the instance directory is `~/.colonymind` unless `COLONYMIND_HOME`
names another).

`colonymind upgrade` does not convert launchd labels from before the name ColonyMind; retire those registrations manually before installing an instance service.

## No memory is recalled

The agent does not use something you told it in another session.

**Check.**

- `colonymind doctor` passes, and `hermes memory status` names `colonymind-memory` as the memory provider.
- In your own session, ask the agent to search its memory for the fact (the `colonymind_memory_search` tool).
  Found there but not used in a reply: recall judged it unrelated to the question, or the question shares no
  word with it while recall matches keywords only (below). Not found settles nothing yet: the search ranks
  with the same selector and calibrated cutoff as recall, so a recorded fact that scores below the cutoff is
  missing from both.
- Whether it is recorded: the unranked read (below) lists the latest messages recorded from a channel, by
  time, with no ranking and no cutoff. Look there for the fact, or, when it was said too long ago to be among
  the latest, for a new message you send on that channel. There: the channel is recorded, and ranking leaves
  the fact out (the cutoff, or keyword recall, below). Missing: it is not recorded, see
  [the owner's turns are not recorded](#the-owners-turns-are-not-recorded).
- Recall is scoped by person: a contact's conversation never recalls what the owner said.

The unranked read is `POST /v1/host/memory/recent`. `person_id` is whose record it reads (yours is
`owner.contact_id` in `colonymind.yaml`), `platform` the channel, and `session_id` one of that channel's Hermes
sessions (`hermes sessions list --source telegram` lists them; the person's own record is read whatever the
session); `limit` is at most 20. Each entry carries the words, when they were said, the conversation and the
`source_id`. `$COLONYMIND_API_KEY` is the instance key (`api.key`).

```sh
curl -s -H "Authorization: Bearer $COLONYMIND_API_KEY" -H 'content-type: application/json' \
  -d '{"identity": {"host_id": "hermes"}, "person_id": "<owner-id>", "platform": "telegram",
       "session_id": "<session-id>", "limit": 20}' \
  http://127.0.0.1:7777/v1/host/memory/recent
```

**Fix.** Follow the cause the checks found: the doctor's remedy, the owner's platforms (next entry) or semantic
recall (below).
With a calibrated cutoff, recall and the search leave out memory that scores below it on purpose, so an
unrelated question gets none ([returning no useful memory](RECALL-HYBRID.md#returning-no-useful-memory)).

## The owner's turns are not recorded

What you say on one channel (the CLI, Telegram, Discord, Slack, Signal and others) is never remembered, or is
remembered as someone else's.

**Check.**

- `hermes logs --level WARNING` shows `turn ... not captured: the sidecar refused it (owner_not_granted) from
  platform ...`: the instance key may not write the owner's turns from that platform.
- `colonymind credentials list` shows the platforms each credential may write the owner from; the Hermes
  adapter writes with the instance key (`api-key`).
- `colonymind people who <handle>` shows whose record the handle you wrote from is on. A handle on a record
  other than yours makes your words that person's.

**Fix.**

- Grant the platform: `colonymind credentials set api-key --owner-platform whatsapp --owner-platform telegram`
  (the list is replaced, so name every platform you write from), then `colonymind service restart`. See
  [managing credentials](CREDENTIALS.md#managing-credentials).
- One of your handles on another person's record: correct the handle (below), never merge the records.
- Merge only a record that is you as well: `colonymind people merge <owner-id> <other-id>` moves every handle
  and the whole history of the other record to yours at once, and whoever writes from any of those handles is
  then treated as you. First check with `colonymind people inspect <other-id>` that every handle on it is yours
  ([relationships](RELATIONSHIPS.md)).

**Correct a handle.** For a phone number, the owner's identity correction moves its phone-format handles
and bare digits on the requested gateway together. A non-phone handle moves individually. If a phone
correction would leave the number identifying two people on any gateway, it returns status 409 with
`identity_phone_ambiguous` and the conflicting `gateway:handle_id:contact_id` values, without recording any change.
Open each named contact and correct or remove that one handle using its address exactly as shown there
and the named contact as `expected_contact_id`. Then retry the original request with the same operation ID.
If correcting a named handle answers `identity_preimage_changed`, correct the number itself on the gateway
named in the refusal (`+` form, the right contact as subject, `expected_contact_id` whoever holds that exact
form there, usually nobody). Then retry the original request: it answers `identity_preimage_changed` if its own
handle has already moved, otherwise it succeeds.
The other person keeps unrelated handles and their history: of what is recorded on their record,
only the sources you list in `source_ids` move to you. List the ones you sent, never theirs (the unranked read
under [no memory is recalled](#no-memory-is-recalled), with their id as `person_id`, lists the latest recorded
on their record, each with its `source_id`). `contact_id` and `subject_id` are your id (`owner.contact_id` in
`colonymind.yaml`), `expected_contact_id` the record the handle is on now, `evidence_refs` your reason, and each
correction takes a new `operation_id` (the same request sent again returns its first result). See
[contacts](SOCIAL-STATE.md).

```sh
curl -s -H "Authorization: Bearer $COLONYMIND_API_KEY" -H 'content-type: application/json' \
  -d '{"contact_id": "<owner-id>", "operation_id": "handle-fix-1", "gateway": "telegram",
       "address": "<your handle>", "expected_contact_id": "<other-id>", "subject_id": "<owner-id>",
       "source_ids": ["<source-id>"], "evidence_refs": ["this handle is mine"]}' \
  http://127.0.0.1:7777/v1/host/social/contacts/correct-identity
```

## Recall matches keywords only

Memory is found only when the question repeats the stored words; a reworded question finds nothing.

**Check.** The `semantic-recall` line of `colonymind doctor`: no `router.embed_url` in `colonymind.yaml` (and
`router.embed_provider` not `local`) means keyword recall only; `mind.faculties.semantic_recall: false` means it is switched off; a failure means the
embedding endpoint is set but not answering.

**Fix.** `colonymind init --embed-url <embeddings-root> --embed-model <model>` records an OpenAI-compatible
embeddings endpoint and turns semantic recall on; where the endpoint has none, `colonymind init --embed-url local`
runs a pinned model inside the sidecar instead (with the `colonymind[vectors]` extra installed): a download of about
1.2 GB into the instance, once. A machine under 8 GB of memory is offered none; name one with `--embed-model`
(`colonymind init --embed-url local --embed-model Qwen/Qwen3-Embedding-0.6B`). When init or doctor's `recall-mode`
says the libraries cannot load the model, the environment has a transformers older than the models need
(transformers 4.51 or later): run the install they name. A switch back to an endpoint, or an uninstall, leaves the
model's files in `<instance>/models`, where they stay until you delete them. Then
`colonymind service restart`. At start the sidecar
queues every recorded message for indexing, so earlier conversations become searchable by meaning as the
queue drains. For a failing endpoint, follow the doctor's remedy
([semantic recall](SOURCE-SEMANTIC-RECALL.md)).

## Recall stopped abstaining after a reranker change

After a change to the reranker, the embedding model or a release that changes what the reranker reads,
unrelated questions get memory again.

**Check.** The sidecar log warns `Rerank abstention calibration mismatch; threshold disabled (current
configuration: ...)`. The calibrated cutoff (`COLONYMIND_RECALL_RERANK_MIN_SCORE`) applies only while
`COLONYMIND_RECALL_RERANK_CALIBRATION` matches the fingerprint of the running configuration: the reranker's
provider, model, endpoint, prompt style and candidate format, the embedding model and its dimensions, and the
weight revision and index generation when set. Any change switches the cutoff off.

**Fix.** Restore the configuration the cutoff was fitted on, or fit a new cutoff on the new configuration
and set both values together in the `environment:` section of `colonymind.yaml`, then
`colonymind service restart`. Copying the new fingerprint without a new fit applies a threshold fitted on
another model. How to fit one: [returning no useful memory](RECALL-HYBRID.md#returning-no-useful-memory).

## Gateway prefetch timeouts

Replies come without memory, and Hermes' log shows `Memory provider 'colonymind' prefetch timed out`.

**Check.**

- `colonymind doctor`: the sidecar is reachable and reports `status=ok`.
- `colonymind service status`: the sidecar service is running and answering.
- The `recall` block of the sidecar's health report (`GET /v1/host/health`) counts each recall that missed its
  reranker by reason (`turns.missed_by_reason`: `budget`, `timeout`, `busy`, `error`).

**Fix.** Start or restart the sidecar (`colonymind service restart`). Recall spends only what is left of
Hermes' wait for the turn, so steady misses point at a slow embedding or reranking endpoint: the doctor's
`semantic-recall` line reports the embedder's latency ([recall latency](RECALL-HYBRID.md#recall-latency)).

## The mind does not act

The agent never reminds you, reaches out or starts work on its own.

**Check.**

- `colonymind mind status`: whether the mind is on, its level, a tripped breaker and the day's budgets.
- `colonymind mind log` and `colonymind mind concerns`: what it considered and what it decided.
- `colonymind mind asks`: what is waiting for your answer.
- `hermes gateway status`: the mind's messages and tasks leave through the Hermes gateway.

**Fix.**

- At `suggest`, the default level, the mind does its internal work, sends you only what you asked for
  (reminders, heads-ups), keeps the rest for the daily digest, and asks before anything reaches a contact or
  the outside. `colonymind mind level standard` lets it message you within its budgets and quiet hours
  ([the mind](MIND.md)).
- Off: `colonymind mind on` clears the off switch; `mind.enabled: false` in `colonymind.yaml` and the level
  `off` stop it too.
- A tripped breaker turns one class of actions into asks until it clears: `colonymind mind reset owner` resets
  the owner class (`colonymind mind status` names the tripped one).
- Budgets (`mind.budgets`) and quiet hours (`mind.quiet_hours`) hold messages until they allow them; a
  faculty set to `false` in `mind.faculties` does none of its work.

## An answered ask is not taken

You answered an ask and nothing happened.

**Check.**

- `colonymind mind asks`: whether the ask is still open. At `mind.ask_expires_hours` (72 hours) an ask lapses with
  no verdict, and an answer after that approves nothing.
- Where you answered. Only your own message in your direct chat with the agent through a gateway is read: never a
  group, a guest's chat, a scheduled run or the terminal, and a yes said on a voice call answers no ask. The
  platform must be one the instance key may write you from (the previous entry).
- What it answered. A reaction or a quoted reply on the ask's own message answers that ask; otherwise your next
  message answers the ask whose message was the last thing in the chat when you wrote. A yes after the agent spoke
  again is conversation, and a code typed in chat answers nothing.
- The words. A hedge, a condition or a question beside the yes makes it unclear, and the agent asks one plain
  question back; an ask that moves money, deletes, changes a credential, messages many people, runs a shell or
  widens what the agent may do without asking takes only a plain yes.
- `colonymind mind asks-report --since 1d`: how asks settled and by whom, and how many clean yeses bound to no ask.
- `colonymind mind why <id>`: what the mind did with the intention, and who answered.

**Fix.**

- Answer the ask's own message: a quoted reply or a reaction, or a plain yes or no right after it.
- `colonymind mind show <code>` sends an open ask to your chat again as its own message; answer that one. The console
  only declines (`colonymind mind no <code>`) or shows an ask again: `colonymind mind yes <code>` approves nothing.
- If your messages are not recorded as yours at all, see [the previous entry](#the-owners-turns-are-not-recorded).

See [asks](MIND.md#asks).

## Memory holds something wrong about a person

Recall shows a claim about someone that is wrong, out of date or not wanted.

**Check.** `colonymind memory inventory <who>` lists what memory holds about the person, newest first: each claim's
state (current, in conflict, superseded, retracted), the words it was formed from and its source.
`colonymind memory history <claim>` shows what the claim replaced and what replaced or withdrew it.

**Fix.**

- A correction beside the original: the annotate route, with the card's source reference and its words as the
  excerpt ([corrections](SOURCE-ANNOTATIONS.md)).
- Removing it: ask the agent to forget it in your own chat, or send the card's `source_id` to
  `POST /v1/host/memory/curated/sources/forget` with `person_id` and `source_ids`; a tombstone suppresses its recall evidence while retaining history
  ([source erasure](RELATIONSHIPS.md#source-erasure)).
- Claims an older extractor formed: `colonymind memory reextract --person <id> --dry-run` prints the plan and its
  estimate; without `--dry-run` the sidecar forms them again in the background
  ([re-extraction](SOURCE-CLAIMS.md#re-extraction)).
