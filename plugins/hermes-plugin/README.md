# ColonyMind adapter for Hermes

The `colonymind` plugin connects stock Hermes to the ColonyMind sidecar through Hermes' own plugin interface (hooks,
a `tool_request` middleware, tools, one command and one prompt section): no patches and no synthetic platform. It
captures each finished turn, reads the owner's answers to asks before the model runs, guards the tool calls of the
mind's own work and of non-owner sessions, and runs the body loop that turns the mind's decisions into Hermes tasks
and messages.
Recall itself comes from the [`colonymind-memory` provider](../colonymind-memory/README.md), shipped in the same
`colonymind-hermes` package.

## Setup

`colonymind init` installs the package into the Python of the `hermes` executable and writes the keys the plugin
needs; `hermes gateway restart` loads it.
`colonymind init --uninstall` removes them and puts back the memory provider init replaced.

```yaml
plugins:
  enabled: [colonymind]
  hook_callback_timeout: 0        # Hermes-wide: plugin callbacks run inline, so no capture is skipped
  colonymind:
    sidecar_url: http://127.0.0.1:7777
    key_file: ~/.colonymind/api.key
memory:
  provider: colonymind-memory
kanban:
  dispatch_in_gateway: true
skills:
  external_dirs: [~/.colonymind/skills]
```

The key file's directory is the ColonyMind instance directory. The plugin reads `colonymind.yaml` (the `mind:`
section) and `identity.yaml` (the owner and the agent's constitution) from it when they are readable; capture and
recall work without them. `colonymind doctor` checks all of this, and
[troubleshooting](../../docs/TROUBLESHOOTING.md) covers the common failures.

In that instance's `colonymind.yaml`, `context.recall_alarm_turns` (default 2)
sets the number of consecutive owner turns missing memory context before the
plugin logs `RECALL_ALARM`. The optional `context.recall_alarm_target` selects
a stock Hermes send target. Only a non-empty string enables delivery; missing,
blank or invalid values keep the alarm log-only. The plugin reads both settings
at startup.

Only the Body loop thread takes the pending notice and sends it through its stock
`send_message` path, outside the counter lock. `flush()`, `tick()` and caller-thread
`run_once()` never send recall alarms. With the Body thread disabled, the alarm
is log-only. There is one pending notice slot; a newer streak replaces an unsent
notice. An `ok` turn re-arms the alarm, with at most one send attempt per streak.
An `ok` before the loop takes the notice drops it; one already taken is still sent.
The text is `Memory context was missing on N consecutive owner turns - exit codes: CODES.`
Codes are sorted and comma-separated. It contains no media tags or turn/session
identifiers. One `RECALL_ALARM_SEND result=sent|failed|uncertain` line reports
the attempt without the target or text. Failed or uncertain sends are never
retried; response hooks only enqueue observations.

## What the model gets

- **One prompt section**, rendered once per session so the prompt cache holds: the owner-authored constitution from
  `identity.yaml` (at most 1,500 characters), the owner's name, the agent's self-narrative (at most 800 characters,
  only in a session that is the owner's alone) and two short notes, at most 4,000 characters in all.
- **Tools**, each checked against who is speaking:
  - `colonymind_self`: the agent's own record (`state`, `log`, `why`, `rate`), its opinions (`opinions`, and the
    owner's `withdraw` and `reconsider`), the owner's standing grants (`grant` files one as an ask, `revoke` narrows
    at once), and the open asks (`asks` lists them in words, `show` sends one to the owner again as its own
    message). No operation answers an ask.
  - `colonymind_people`: `who`, `inspect` and `propose_link`; the owner's `set_permission`, `set_cadence` and `merge`.
  - `colonymind_memory_search`: search the person's own evidence beyond this turn's recall, with speaker, time and
    source. `window` (`{start, end}`) names the span a question asks about, whatever words it uses, so that span's
    messages are read even when they share no word with the question; `as_of` is the time relative dates count
    from. Both are ISO 8601 times with an offset, sent in UTC; a time the sidecar would refuse is not sent, and the
    model is told the valid form. `colonymind_memory_forget` (owner only): tombstone sources the owner asks to forget while retaining history.
    Its compatibility key `source_erased` means a tombstone was recorded; the result includes a durable receipt.
  - `colonymind_reminder`: a reminder from a recalled deadline, scheduled on stock cron.
  - `colonymind_secret` (owner only): use a sealed value by reference, without the model ever seeing it.
  - `colonymind_action`, only when the owner declared actions in `colonymind.yaml`: a fixed command with checked
    arguments.
  - In a group chat the reply goes to everyone, so for every sender, the owner included, a tool reads that chat's
    own history only: a memory search and a page are narrowed to it, `who` and `inspect` say who one named person
    is, and `session_search`, the opinions and a rating are refused. Such a result says so in one line
    (docs/HERMES-ADAPTER.md, Owner and guest).
- **`/mind`** for the owner in chat: `status`, `log`, `why <id>`, `asks` (in words, never by code) and `off`. Every
  other change goes through the `colonymind` CLI.

## Answers to asks

The owner answers an ask as they would answer a person: a plain yes or no right after the ask's message, a quoted
reply to it, or a reaction on it; where the ask's own message offers it, "always" also stops the asking for exactly
that kind of work, until the owner takes it back (`colonymind mind allows`). Before the model runs, the plugin hands the owner's own message to the sidecar with
what only the host can see: when it arrived, its id, what it replies or reacts to, and what reached the chat before
it. The sidecar decides what the message answers; the model is told what it did, so it acknowledges once and never
asks again. The words are not kept at that point, only a salted fingerprint to match the message to its turn. Only
the owner's own direct chat through a gateway is read; a guest, a group, a scheduled run, a worker and the local
command line post nothing. A tap that answers an ask takes no model turn.

## Guard and body

- **Guard.** Every tool call of a mind-originated run (a kanban worker on the `colonymind-act` profile) and of a
  non-owner session passes `pre_tool_call`. The floor is decided by what a call does (the tool and its target),
  with the same effect table the sidecar uses; effectful calls fail closed when the sidecar is silent; writes stay
  inside the task workspace; and in a contact's session a refusal is worded for the contact, never an approval
  prompt they could answer.
- **Secrets.** A secret in a gateway message is sealed before Hermes sees it, so Hermes' transcript, the model
  provider and session search never receive the value; a tool call gets it filled in only where the owner's own
  latest words stand behind it.
- **Body.** One thread per Hermes process delivers the durable turn outbox, then runs the mind loop: it creates one
  kanban task per intention the mind dispatches (idempotency key `mind:<id>`), sends the mind's messages once
  through stock `send_message_tool`, reports each task's outcome and what the board shows (stale, blocked and goal
  tasks) back to the sidecar. A send begun but never confirmed is reported as uncertain and never repeated. With the
  mind off, the mind's tasks and messages are not dispatched or sent.

## Modules

| Module | Job |
| --- | --- |
| `__init__.py` | Registration: hooks, the `/mind` command, the tools and the prompt section; starts the body thread |
| `client.py` | Settings, bearer key, short timeouts and a circuit breaker |
| `capture.py` | The session map (who is speaking) and the durable SQLite turn outbox; hooks only enqueue |
| `answers.py` | The owner's answers to asks, read before the model runs, and the ledger of what reached each chat |
| `body.py` | Turn delivery and the mind loop: dispatch, outbox, outcomes, the board's state |
| `guard.py` | `pre_tool_call` rules for mind-originated and non-owner runs |
| `secrets.py` | Sealing at the gateway, fills and masks on tool calls, the owner's `colonymind_secret` tool |
| `commands.py` | `/mind` and the gateway gate that keeps it the owner's |
| `tools.py` | `colonymind_self`, `colonymind_people`, the memory tools and declared actions |
| `reminders.py` | `colonymind_reminder` on stock cron; the plugin keeps only the job id |

Besides the public plugin API, the plugin imports a few Hermes modules directly: `hermes_cli.kanban_db` (dispatch,
reconciliation, the board's state), `tools.send_message_tool` (the outbox), `cron.jobs` and `cron.scheduler`
(reminders) and, optionally, `agent.prompt_builder.clear_skills_system_prompt_cache` (without it a new skill shows
after a restart). The adapter tests run these against stock Hermes.

See [the Hermes adapter guide](../../docs/HERMES-ADAPTER.md) for every hook and route.

## Work records

The `work_records` faculty is on by default. It records bounded operation facts
from genuine owner turns without asking a model. It remains active with the mind
or lessons off. The body thread refreshes its cached faculty setting even when
the mind is off; disabling it clears pending in-memory capture. The sidecar
checks owner identity and credential grants before admitting a root.

Tool hooks only update bounded memory. The existing final-turn hook writes the
frozen work section in the durable TurnOutbox before delivery. That durable
commit retains its existing disk cost. New tool metadata hooks have a separate
offline p99 budget below 1 ms. Child bindings use the native parent turn ID and
later child results use independent durable work events.

Paths and other exact resource values stay only in bounded transient lookup
tables. Stored resource references are opaque slots, scoped to the root and
execution task. Selected receipt IDs pass the existing secret detector before
outbox hashing. Raw arguments, result text and content hashes are not copied
into work records. Stock file writes can supply their native post-write hash
attestation. A stock patch without that explicit attestation remains unproven.
Saved-job verification needs a later read containing every requested field;
display previews or missing fields cannot certify a job.

Version 1 turn delivery is the default. A queued `transport_version: 2` uses
`PUT /v2/host/turns/{id}` with the same work section. A successful turn response
must explicitly acknowledge the work section. An old sidecar that ignores it
or refuses a work event leaves a visible refused outbox row. A missing parent is retried.
See [the work contract](../../docs/WORK.md) for statuses, lineage and erasure.
