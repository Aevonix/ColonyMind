# Hermes adapter

**A mind for Hermes agents: persistent memory, commitments, initiative and measured autonomy.**

`colonymind-hermes` connects stock Hermes (`hermes-agent >= 0.21.3, < 0.24`, no patches) to the ColonyMind sidecar,
which runs as a tool of its own. `colonymind init` installs the matching version of this package into the Python of
the `hermes` executable; installed there first, it also lets `hermes memory setup` run init. It is one wheel with
three Hermes plugins:

- `colonymind` (general): turn capture, the owner's answers to asks read before the model runs, the guard, `/mind`,
  and the self, people, memory, reminder and secret tools
  ([its README](https://github.com/Aevonix/ColonyMind/blob/main/plugins/hermes-plugin/README.md))
- `colonymind-memory` (memory provider): per-turn recall through `/v1/host/context/assemble`, scoped to the person
  speaking, turn sync when the general plugin is absent, and the owner's commitment write
  ([its README](https://github.com/Aevonix/ColonyMind/blob/main/plugins/colonymind-memory/README.md))
- `colonymind-gates` (optional, off until configured): a cron wake gate, inbound triage and a tool-result screen,
  measured in shadow before they act ([Hermes gates](https://github.com/Aevonix/ColonyMind/blob/main/docs/HERMES-GATES.md))

What ColonyMind does, the evidence against stock Hermes and what is not yet demonstrated:
[the project README](https://github.com/Aevonix/ColonyMind/blob/main/README.md). Every guide:
[the documentation index](https://github.com/Aevonix/ColonyMind/blob/main/docs/README.md).

## Install

```bash
uv tool install --python 3.12 colonymind && colonymind init && hermes gateway restart
```

`colonymind init` installs `colonymind-hermes` into the Python of the `hermes`
executable, writes the config keys below and creates the `colonymind-act`
worker profile. `colonymind upgrade` repeats the adapter install when the
version changed. Both record the sidecar and adapter versions in
`<instance>/hermes-install.json`, and init records there the memory provider
it replaced, which it names in its output and `colonymind init --uninstall`
puts back.

The other front door is Hermes' own: `hermes memory setup`, then
`colonymind-memory`. The provider's `post_setup` runs in Hermes' environment,
where the sidecar package is not installed (the adapter and the sidecar are two
packages), so it runs the same `colonymind init` as a subprocess, with the
instance the profile is attached to (`--home`: the configured key file's
directory, else `$COLONYMIND_HOME`, else `~/.colonymind`, the rule the provider
reads it by), the profile's home and Hermes' interpreter (`--hermes-home`,
`--hermes-python`), and init asks its own questions. When no `colonymind` command is found outside
Hermes' environment, it offers to install the sidecar as a tool of its own
(`uv tool install --python 3.12 colonymind==<adapter version>`, else pipx); it
never installs anything into Hermes' environment, and the provider's manifest
declares no pip dependency for stock setup to install first. Init then
installs the adapter into Hermes' interpreter by the one mechanism it always
uses (the installed package's entry points); a provider directory under the
profile's plugins, which Hermes loads ahead of those, is named by setup. The
setup schema (`get_config_schema`) offers only the connection: the sidecar URL
and the key file.

```yaml
plugins:
  enabled: [colonymind]
  hook_callback_timeout: 0          # callbacks run inline; overlapping fires are not dropped
  colonymind:
    sidecar_url: http://127.0.0.1:7777
    key_file: ~/.colonymind/api.key  # one line, mode 600
memory:
  provider: colonymind-memory
kanban:
  dispatch_in_gateway: true         # the plugin's heartbeat is the dispatch tick
skills:
  external_dirs: [~/.colonymind/skills]   # ColonyMind's own skills (with faculties.skills on); nothing else writes there
security:
  protected_instruction_extra_patterns: [colonymind.yaml, identity.yaml, api.key]
```

Every request to the sidecar carries `Authorization: Bearer <key>`. The key is bound to the contacts and
channels it may write as ([credentials](https://github.com/Aevonix/ColonyMind/blob/main/docs/CREDENTIALS.md)): it writes a turn as the owner only from the
platforms `colonymind.yaml` `credentials.api-key.owner` lists (init writes the terminal, `cli`, and the
platforms of the owner's handles, in Hermes' platform names).

## Seams used

| Need | Host integration |
|---|---|
| Who is speaking | `pre_llm_call` (`session_id`, `platform`, `sender_id`, `user_message`), kept in a bounded session map |
| Turn capture | `post_llm_call` enqueues one row in `<hermes_home>/state/colonymind-turn-outbox.sqlite3`; the body thread delivers it to `POST /v1/host/turns/sync` |
| Native turn settlement | On stock Hermes, the adapter wraps `TurnFacadeMixin.run_conversation` and `conversation_loop.run_conversation` in process; when the host provides `on_native_turn_settled`, it uses that hook instead. Both paths capture failed or interrupted turns that skipped `post_llm_call` and release an interrupted turn's durable transport binding before an injection |
| Delegated branches | the child's final `post_llm_call` enqueues its message list, tool results, delegation brief and summary for `POST /v1/memory/branches`; an owner-only `colonymind_consult` reads named branches together |
| The owner's answers | `pre_gateway_dispatch` hears each gateway message first (its arrival time, its id, what it replies or reacts to, and a salted digest of its words, never the words) and reads the owner's reaction on an ask at once; `pre_llm_call`, registered after capture, hands the owner's own message to `POST /v1/mind/answers` before the model runs and gives the model what it did as context (Asks, below) |
| Chat ledger | what reached each owner chat, for the answer route (kinds, times and ids, never words): the body's sends, the reply (`post_llm_call`), interim messages (`on_interim_message`), the model's own `send_message` (`post_tool_call`) and every cron run's reply |
| Verified work | bounded `post_tool_call` facts, model-call attribution and failures (`pre_api_request`, `post_api_request`, `api_request_error`) and `subagent_start` lineage through `parent_turn_id`; frozen turn envelopes and separate late events use the durable outbox, with verification in the sidecar ([contract](https://github.com/Aevonix/ColonyMind/blob/main/docs/WORK.md)) |
| Reactions | WhatsApp reactions and the tapbacks a tapback adapter delivers arrive as gateway messages; Telegram reactions through `gateway_platform_event` |
| Invented codes | `transform_llm_output`, in the owner's own chat: a line of the reply that tells the owner to answer with a code is removed (best effort: Hermes takes the first transform that returns text) |
| Recall | memory provider `prefetch` |
| Recall indicator | memory provider `recall_status`: "recalled N" (N the quotations the memory section cites) for the owner's private turns only; Hermes shows a status line outside the model's context by default, or the provider dispatches a routed notice in the background |
| Availability | memory provider `is_available` and `unavailable_reason`, from configuration and files alone (no network) |
| Backup | memory provider `backup_paths`: the instance directory, for `hermes backup` |
| Setup | memory provider `post_setup` (runs `colonymind init`) and `get_config_schema` (the connection only) |
| Guard | `pre_tool_call`; a block directive fails closed, a floor match becomes an `approve` directive (Hermes' human-approval gate, which fails closed in a worker) |
| Heartbeat | `on_kanban_dispatch_tick` wakes the body thread; stock fires it from `dispatch_once` in the one process holding the dispatcher lock, so it also marks this process as the board's writer |
| Pause | `agent.estop.is_engaged()` (`hermes pause`): the body creates and sends nothing while it holds |
| DM lookup | the stock session rows (`hermes_state_registry`, `list_gateway_sessions`): the DM chat the gateway has had with a sender, for handles that name a user rather than a chat (Discord) |
| Durable work | `hermes_cli.kanban_db.create_task(idempotency_key="mind:<id>", assignee="colonymind-act", ...)`; an existing non-archived task with the key is returned, never duplicated |
| Messages | `tools.send_message_tool.send_message_tool({"action": "send", "target", "message"})`, text verbatim |
| Task outcomes | the `mind:*` task's status plus its latest `task_runs` row (`kanban_db.latest_run`) |
| Owner commands | `register_command("mind")`: read-only in chat plus `off`; `pre_gateway_dispatch` drops a `/mind` whose sender is not the owner, since the command handler gets no sender |
| Tools | `register_tool`; handlers receive `session_id`, which the session map resolves to a sender |
| Reminders | stock cron: a one-shot job whose id is the only thing the plugin keeps |
| Skill loads | `on_skill_lifecycle`: a `loaded` fact for a `colonymind-*` skill is queued in the turn outbox as a `skill_use` row, which the body delivers to `POST /v1/mind/skills/used`; every other skill and action is Hermes' own |
| Skills index | `skills.external_dirs` lists `<instance>/skills`; when `/v1/mind/state` reports a new `skills.generation`, the body calls `agent.prompt_builder.clear_skills_system_prompt_cache()` so the next session lists the change (stock caches the index per process with no file times in its key); without that function a change shows after a restart |
| Secrets | a second `pre_gateway_dispatch` seals a gateway message's secrets (and the quoted message and backfill that come with it) before Hermes sees them; `register_middleware("tool_request")` serves the owner's sealed values to their own calls (into a send's or a write's words, or a command's or code's environment for that call) or hides them in a worker's; a second `pre_tool_call` holds a call whose value waits for the owner or cannot be used there; `post_tool_call` removes a call's private value files unless the program that took one still runs; `register_redaction_patterns` and `agent.redact.register_vault_redaction_value` keep used values out of model-facing output; `colonymind_secret` (see Secrets) |

## Owner and guest

The owner is the sender whose handle is listed in `identity.yaml`
(`owner.handles.<platform>`) or whose resolved contact is `owner.contact_id`.
Turns without a sender on the CLI, cron or API lanes are the owner's own for
the plugin's tools; a cron run (`platform: cron`) is a stored prompt rather than the
owner typing, so it cannot answer an ask, rate, change a permission or
forget. In memory a turn is the owner's only when it arrives on a platform the key
is granted the owner on: every turn names its platform (the sender's, the lane
itself: `cli`, `cron`, `api`, or the platform the memory provider was initialized
on when no session binds one). A turn the sidecar refuses (`owner_not_granted`) is
never sent again as another contact: the adapter logs it and counts it as a failed
capture (the outbox row keeps the refusal and `/mind status` counts it; the memory
provider's diagnostics count `attribution_refusals`). Grant the CLI with
`colonymind init --owner-platform ...` or `colonymind credentials set api-key
--owner-platform ...` to keep a CLI conversation as the owner's. Any other sender is a guest: `session_search` is blocked for guests,
delivering cron jobs need a recipient whose `may_contact` is not `never`,
every mutation through `colonymind_self`, `colonymind_people` and
`colonymind_memory_forget` is refused, `colonymind_people who` without a name is
refused (a guest may ask who one named person is, never list the contacts), and
the plugin drops a guest's `/mind` at `pre_gateway_dispatch`, because Hermes
hands the command itself no sender.

A guest's tools are the guest tools. Hermes gives its tools per platform, not per sender, so a contact on a platform
the owner also uses (and an API or voice client that sends no owner identity) is offered that platform's terminal,
code and file tools. In a guest's session the guard lets through only:

- the guest tools: `web_search`, `web_extract`, `x_search`, `vision_analyze`, `video_analyze`, `image_generate`,
  `text_to_speech`, `clarify`, `todo`, `todo_list`, `skills_list`, `skill_view`, `tool_search`, `tool_describe`,
  `read_file`, `search_files` and the messaging tools under the rules above (`session_search` stays refused);
- the ColonyMind tools, each of which answers a guest itself, but `colonymind_action` (the owner's declared actions);
- a tool the operator names in `guests.tools` (an MCP server's search, say), never one of those below; the plugin
  does not know its arguments, so every string it is given that names a local file (a path from the root or the
  home, a `file://` link, a relative path, a name that exists where it would look) must be a file the sender sent.

Every other tool is refused before any sidecar verdict, as the plugin's one final answer ("... is not available in
this conversation"), with no sidecar needed: what runs a program or code (`terminal`, `process_manage`,
`execute_code`, `delegate_task`, `computer_use`, the desktop panes), every `browser_` tool (the vault's among them),
what writes files or the agent's shared memory (`write_file`, `patch`, `skill_manage`, `memory`), what runs later in
the owner's name (`cronjob_manage`, `kanban_create`, `kanban_comment`, `kanban_unblock`, `colonymind_action`), and any
tool the plugin does not know (a plugin's, an MCP server's). A file a guest tool takes must be one the gateway saved
from that sender's own messages in that chat (the local path its own event carried for an attachment, links followed;
a backend that sees the cache elsewhere may name it there; a delegated child's chat is its origin's, and a session no
gateway bound is the sender's direct chat): a `read_file` or `search_files` path, the image or video
`vision_analyze` and `video_analyze` look at, the image `image_generate` edits and each of its reference images, and a
`MEDIA:` attachment or a media file in a message. A path anyone typed (however it is worded), another person's file
in the same cache, the sender's own file from another chat, a directory and a relative path are refused ("... reaches
only the files sent in this conversation"), and `text_to_speech` writes only where Hermes puts its files.

An operator who wants otherwise opts in, in `colonymind.yaml`:

```yaml
guests:
  executor_platforms: [api_server]   # these platforms' non-owner sessions keep every tool (default: none)
  tools: [mcp__docs__search]         # extra tools a non-owner session may call anywhere (default: none)
```

Each counts only as a list of non-empty names; anything else opts nothing in. A platform in `executor_platforms` (a
local API lane whose every caller holds its key, say) keeps every tool, but the instance key and the answer routes
stay out of reach there too (the tripwire, below); a shell can spell a name in more ways than any rule sees, which is
why nothing is opted in by default and `colonymind doctor` warns for an opted-in platform (`guest-lane-tools`, with
its remedy: remove it, or run Hermes' tools on a container or remote backend).

Who a session is follows where it comes from. A delegated child (stock runs one on the `subagent` platform with no
sender) and a session the plugin has not seen a turn of (a compression's new id) take the standing of the session
they come from, its parents followed in this process or in Hermes' session store: the owner's child keeps the
owner's tools, a guest's child is a guest. When the session store cannot be read and no sender or creating-turn
standing was recorded, the session is unknown and never borrows the gateway's bound sender.
With no parent to follow, the gateway's binding of the current turn says:
its sender is the owner only by the owner's handle or resolved contact, and a guest otherwise, a sidecar that cannot
answer included; a gateway platform with no sender is a guest. With nothing at all the session is unknown and the
guard refuses every tool with the guest-style final refusal. Unknown sessions get no guest tools: without a verified
sender or registered parent even a read could expose another conversation's files or shared memory. This includes
every MCP name (`mcp__*` and legacy `mcp_*`), browser vault tools and tools added by plugins or guest opt-ins. A
declared action (`colonymind_action`) runs only for a session established as the owner's, or in a kanban task.

Standing is per turn in a group. Only a turn whose current sender is verified as the owner gets owner standing;
an earlier owner turn never upgrades a guest's next message in the same group. Binding caches include the current
sender, platform and chat. Delegation snapshots the creating turn before the child can use tools, so a later group
speaker cannot change that child's standing. The host-private standing store retains these snapshots across restarts.
Hands also retain the snapshot in their admission spool; registration alone grants no standing. Successful cron
creation or update records the caller's standing, and each run records that origin when observed. An unresolvable
cron origin is refused, including reads; it never enters the mind lane. A registered mind worker records mind
standing from the dispatcher's task/profile binding. Observed terminal and system roots record their explicit host
standing; an unregistered parentless session remains unknown.

Unknown refusals emit a WARNING with fixed reason `standing_unknown`, tool name and a fixed session class. They
never log session, contact or sender ids, arguments or exception text. Guest tools and explicit guest opt-ins retain
their existing policy for established guests.

## Mind-originated runs

A kanban worker spawned on the `colonymind-act` profile is a mind-originated
run (the dispatcher sets `HERMES_KANBAN_TASK` and `HERMES_PROFILE`). In it:

- `kanban_create` only with `assignee: colonymind-act`, no `mind:` key, no
  project override and no workspace outside `HERMES_KANBAN_WORKSPACE`
- `write_file` and `patch` only inside `HERMES_KANBAN_WORKSPACE`, every V4A
  header target included
- `mind.deny.tools` and `mind.deny.text` from `colonymind.yaml` block; floor
  patterns (money, irreversible deletion, credentials, bulk messaging) ask
- messaging tools and delivering cron jobs need a permitted recipient: the
  plugin resolves a job's effective `deliver` and `failure_deliver` contacts
  (a stored job's when an update does not resend them) and hands them to the
  sidecar as `recipients`, which applies `may_contact` and the message budgets
  to each of them exactly as it does to a `send_message` recipient
- with `mind.enabled: false` or `autonomy: off`, every effectful tool is blocked
- a call that names the instance key or an answer route, or a read that reaches the key file however it is spelled,
  is refused (the tripwire, below)

The guard asks `POST /v1/mind/guard {tool, args, session, run, task_id, owner,
contact_id, platform, sender_id, recipients}` with a 2 s timeout when the
sidecar serves the mind routes and reads `{allow, reason}` (`ask: true` becomes
Hermes' approval gate in the owner's and the mind's runs); silence blocks effectful tools, a 404 falls back
to the rules above. Read-only tools never wait on the sidecar.

In a contact's (non-owner) session nothing becomes Hermes' approval gate: in a gateway that gate posts its
prompt to the session's own chat and waits for `/approve` or a bare "yes" from it, so the contact would be
the one approving. There an `ask` verdict is refused, and every refusal is the plugin's one final answer
(`retry: false`), worded for a reply the contact reads. The final response is delivered to the session's
own chat and nowhere else: the chat the gateway bound for the turn (a group's chat in a group), or, with no
gateway, the sender's direct chat. A message to that chat is answered with "your final response is
delivered to this conversation as your reply": the reply already is that message. Anything else, another
handle of the sender's and a direct message to the sender from a group included, is a recipient for the
sidecar's verdict; a target no contact is known at is answered "nothing was sent: no contact is known at
that target".

## The body: the mind's effects on Hermes

The body thread runs in every Hermes process, but only the one that owns the
kanban dispatcher writes to the board: stock fires `on_kanban_dispatch_tick`
from `dispatch_once` in the process holding the singleton dispatcher lock, and
the body runs steps 2 to 5 only after it has seen that tick. A CLI session, a
gateway that lost the lock and a kanban worker drain the turn outbox only; the
first two also read `GET /v1/mind/state` on every tick for the skills
generation (step 1), since Hermes caches the skills index per process. While
`hermes pause` holds (the stock ESTOP sentinel) steps 2 and 3 wait as well. Once
the sidecar serves `/v1/mind/*`, the body does this on every tick (about every
60 s, and at once when the dispatch tick or a captured turn wakes it):

1. `GET /v1/mind/state`. With `enabled: false` there, or `mind.enabled: false`
   in `colonymind.yaml`, steps 2 and 3 are skipped and every unstarted `mind:*`
   task is archived; running workers end on their own. None of this needs a
   model endpoint. When `skills.generation` moved since the last tick (the
   sidecar wrote or removed one of its skills), Hermes' skills prompt cache is
   cleared; the first value a process sees is only recorded.
2. `GET /v1/mind/dispatch` → for each `kind: task` intention one
   `kanban_db.create_task(title, body, assignee="colonymind-act",
   idempotency_key=dedup_key or "mind:<id>", workspace_kind="scratch",
   created_by="colonymind", priority, max_runtime_seconds, max_retries,
   goal_mode, goal_max_turns)`, then `POST /v1/mind/dispatch/{id}/bound
   {hermes_ref, hermes_kind: "kanban", status, bound_at}`. A lost ack or a
   restart creates no second task: the key returns the existing one, and a
   task with the key that was archived in between (stock's lookup skips
   archived tasks) is bound and settled as `cancelled` rather than recreated.
   Messages and notices never travel through dispatch. Runtime and retries
   are the task kind's (`mind.budgets.task_types`, else
   `task_max_runtime_s` / `task_max_retries`), and fall back to that flat
   pair when a payload carries none.
3. `GET /v1/mind/outbox` → for each message, in order: skip it if the body's
   own ledger has begun it before; `POST /v1/mind/outbox/{id}/sending {target,
   at}` (any non-2xx: not ours); record `sending` locally; send **verbatim**
   with `send_message_tool`; record the result; `POST
   /v1/mind/outbox/{id}/sent {result: sent|failed|uncertain, error, hermes_ref,
   at}`. `success` from the tool is `sent`; an error raised before the platform
   call (unknown target, unconfigured platform, no home channel) is `failed`;
   `Send failed: ...` or an exception is `uncertain`. A
   process that stops between `sending` and `sent` reports that message
   `uncertain` at the next start and never sends it, whatever the sidecar
   lists. The target is the message's `target` (`platform:chat_id`), else
   `recipient.platform` + `recipient.chat_id|address|handle`, else, for the
   owner (`recipient_is_owner`, the owner contact or no recipient), the first
   handle in `identity.yaml` `owner.handles` (the `[{platform, id}]` list
   `colonymind init` writes, or a `{platform: [ids]}` mapping), else the first of
   the sidecar's `recipient_handles` (`{gateway, address}`, primary first). A
   handle names a sender; where that is not a chat (a Discord user id) the
   target is the DM session stock has recorded with that user, and the handle
   itself when none has been seen yet.
   The sidecar answers `sending` with 409 for a message it already handed
   out, and marks a claim nobody settled `uncertain` after ten minutes. A
   claim that names no target (`target: null`, the body resolved no handle)
   is also 409 (`no_target`) and is not a claim: the message stays ready,
   never `failed`, and the next pull offers it again with `recipient_handles`
   read afresh, so a handle added later lets it go out.
4. Reconciliation: for every `mind:*` task, `GET /v1/mind/why/{id}`. A 404, or
   a lifecycle `status` in `proposed | asked | denied | expired | cancelled |
   dropped`, means no dispatched intention owns the task and it is archived
   (the ask backstop of architecture 7.7). Otherwise, when the task is
   `done | blocked | review | archived` or its latest run has ended, one
   `POST /v1/mind/outcome {id, hermes_ref, hermes_kind, status, outcome:
   done|blocked|failed|cancelled|uncertain, final, summary, error, verified:
   "hermes_failure"|null (the sidecar honours the claim only for a failure
   with a reason, and never takes `owner` or `check` from the body),
   run: {id, outcome, status, profile, started_at,
   ended_at}, block_kind, consecutive_failures, completed_at, observed_at}` per
   state change (a failed run that Hermes requeues is one `failed`, `final:
   false` outcome; a task parked `blocked` with its failed run once
   `consecutive_failures` reached `max_retries` is `failed`, `final: true`; a
   `kanban_block` is `blocked`; a 404 archives the task). The body remembers
   what it posted in `<hermes_home>/state/colonymind-body.sqlite3` so an outcome
   is posted once.
5. `POST /v1/mind/observations {observed_at, board, body: {pid, profile,
   started_at, interval_s, ticks, mind_ticks, last_tick_at, last_mind_tick_at,
   last_pull_at, stale}, counts: {status: n}, stale_tasks, blocked_tasks,
   goals, mind_tasks}` when the board changed and at least every 5 minutes.
   Task entries carry `id, title, status, assignee, created_by, created_at,
   started_at, age_s, idle_s, block_kind, goal`; a stale task is an open
   non-mind task idle for `mind.stale_task_hours` (default 72); goals are
   kanban goal-mode tasks; `mind_tasks` add `intention_id` and the latest run.

Every `GET /v1/mind/dispatch` is the body's heartbeat: a sidecar whose last
pull is older than 5 minutes stops deliberating (architecture 3.1).

`colonymind_hermes.tick()` runs one full tick on the caller's thread: `POST
/v1/mind/tick` (the sidecar forms intentions now) and then the body pass above.
The paired benchmark's body tick calls it before Hermes cron and kanban
dispatch, with `COLONYMIND_BODY_THREAD=0` so the body's own thread stays
parked and nothing lands between two observed ticks; a gateway never needs
either.

## Asks

An ask lives only in the sidecar and creates nothing in Hermes until approved. The owner answers it in their own
chat: the sidecar decides which ask a message answers and what its words say (docs/MIND.md, Asks), and the plugin
brings it what only the host can see.
Hand ask messages carry the controller's fixed review deadline. The body waits
for the matching native parent review to end, including silent settlement, and
then the 5 second reply gap. Prior tool work does not change this ordering.
The deadline is the earlier of 3 minutes after the hold and the hand's wall
deadline, so missing completions and hung reviews cannot block delivery forever.
The console only declines an ask (`colonymind mind no <code>`) or shows it again, and `colonymind mind yes`
approves nothing; the 1.11 route that took a code and the owner's words from the request body is removed.

- **Hearing.** `pre_gateway_dispatch` records each gateway message per sender, before Hermes adds to it: when it
  arrived (a message queued behind a busy session keeps its own time), its id, what it replies or reacts to (a
  WhatsApp reaction, a tapback adapter's reaction), a voice note, the agent's own account or a forwarded mark, and a
  salted digest of its words. Never the words: a secret in them is sealed only after this hook. A turn takes the
  entry its gateway bound it to, else the one whose digest matches its own text; two quick messages never swap what
  they reply to, and where two look alike (two voice notes with no caption) neither is read. A message steered into
  a running turn carries none.
- **Ingress.** Only a genuine inbound message is read as the owner's answer, before the model or at ingest: its turn
  took the gateway's own record of it, the record carries the platform's message id, and the words captured are still
  the words the record matched. Hermes never runs `pre_gateway_dispatch` for an internal event, so a background
  delegation's completion, a redirect or steer line (even one joined to the owner's message) and an injected or
  synthetic turn are never read as an answer, whatever their words; such a turn's capture names no message. With
  hands installed, the front's ingress ledger also vouches for a turn it bound to the owner's message by its platform
  id (`owner_started_turn`), and the words read are the front's, with the redirect lines it added removed.
- **Reading.** In the owner's own direct chat through a gateway (never a guest's, a group, a kanban worker, a cron
  run, a delegated child or the CLI), `pre_llm_call` posts the owner's message to `POST /v1/mind/answers` when an
  ask is open (the sidecar's count, at most 30 s old, read again when something reached the chat since) or the
  message replies or reacts to something: the platform, the sender, the chat, the turn, the owner's own words after
  the secret seal, the message's own evidence and the chat's ledger as of its arrival (at most 100 entries of the
  last 48 hours). The read has 1.5 s in all for every call it makes. Past that, or for a message over 8,000
  characters, the model runs without it and the sidecar reads the captured turn at ingest (the evidence rides in the
  turn's `metadata.answer`), by a reaction or a quoted reply only. The route's `context` reaches the model before it
  replies, so it acknowledges an approval once and never asks again.
- **Taps.** The owner's reaction in their direct chat is posted at `pre_gateway_dispatch`, within the same 1.5 s;
  when it bound to an ask, or settled or undid one, the event is skipped (a tap takes no model turn) and the owner
  gets one line from the outbox. Telegram reactions come through `gateway_platform_event`.
- **The ledger.** What reached each owner chat, in `<hermes_home>/state/colonymind-chats.sqlite3`, shared by every
  process of the profile (kinds, times and ids, never words): the body's sends with their outbox and delivery ids,
  in the order it sent them; the turn's reply (`post_llm_call`); interim messages (`on_interim_message`); the
  model's own successful `send_message` (`post_tool_call`: a send to the owner counts in every owner chat on that
  platform); every cron run's reply (in every owner chat, since where a run delivers is not visible here); and a
  turn under way, which counts as having spoken just before a message that arrives during it. What an entry showed
  is the sidecar's own record, never the plugin's word.
- **After the reply.** An ask notice, a shown ask and a plain question (`ask_notice`, `ask_show`, `ask_clarify`)
  wait while a turn in the target chat has no reply and 5 s after it (`Body.send_outbox` asks `Answers.hold`), so they
  land last and apart; a turn with no reply after 10 minutes holds nothing. A background ask needs no extra wait
  when no turn is open and no reply is still settling in that chat.
- **No approving tool.** `colonymind_self` has no `yes` or `no`; a model that still calls them is told that the
  owner's own message answers an ask, and what this turn's message did. `asks` lists the open asks in words (the
  owner's own chat only) and `show <n>` has the sidecar put one in front of the owner again as its own message
  (`POST /v1/mind/asks/{id}/show`), in the sidecar's own words with every term a yes grants; the plugin never words
  an ask, and a sidecar that cannot show one shows nothing. `/mind asks` and `/mind status` show asks in words, and
  `transform_llm_output` removes from the reply, in the owner's own chat, a line that tells the owner to answer with
  a code (logged as `invented_ask_reference`).
- **The tripwire.** In every session, the owner's own included, the guard reports a tool call that reads the
  instance key or names an answer route (`POST /v1/mind/tripwire`, once per signal and session in 10 minutes), and
  the owner is told at most once a day. In the owner's own lane, and in a session nothing tells about, the call runs:
  an approval never stands in the owner's way. In a guest's session (an opted-in platform's included) and in a mind
  run the call is refused as well, and there a read that reaches the key file however it is spelled counts too: a
  `read_file` whose path resolves to it (a link, another case) and a content `search_files` with any root that is
  the key file or a directory holding it (each part of a path of several, which stock searches one by one) whose file
  glob could match it read as ripgrep reads one (`{a,b}` alternatives, `**/`, escapes, `[^...]`, any case); a file
  named as the root counts whatever the glob says, since the search reads it anyway.

`colonymind_self rate {id, verdict}` posts `POST /v1/mind/rate` for the owner only. `log` and `why` are the owner's
too, and a guest's `state` says only whether the mind is on and at what level: the log and the asks name what the
owner asked about other people.

## People

`colonymind_people` is the model's view of the people store over `/v1/mind/people`
(the sidecar side is `sidecar/colonymind/api/routers/people.py`):

| Operation | Who | Route |
|---|---|---|
| `who {contact_id}` | everyone (listing: the owner) | `GET /v1/mind/people?q=` (a name, handle or id; empty lists the newest, for the owner only) |
| `inspect {contact_id}` | everyone | `GET /v1/mind/people/{who}` |
| `propose_link {contact_id, handle: gateway:address}` | everyone | `POST /v1/mind/people/link`: a candidate the owner confirms as an ask |
| `set_permission {contact_id, permission}` | owner | `POST /v1/mind/people/{who}/permission {may_contact: never\|ask\|auto}` |
| `set_cadence {contact_id, minutes}` | owner | `POST /v1/mind/people/{who}/cadence {minutes}` (0 clears) |
| `merge {contact_id, drop}` | owner | `POST /v1/mind/people/merge {keep, drop}`: `drop` folds into `contact_id`; handles, sources, comms and affect follow the person |
| `boundary {contact_id, text?, drop?}` | everyone (a guest: the page about themselves) | `GET /v1/mind/pages/{who}` shows the agent's page on the person, each line with its id; with `text` or `drop`, `POST /v1/mind/pages/{who}/lines {kind: boundary, text, drop, source}` adds a boundary that names the turn it was written in, or removes one by its line id |

A guest sees who someone is (`contact_id`, `display_name`, `trust_tier`) and
nothing else: the plugin names the guest as the viewer (`contact_id`) on its
reads and the sidecar answers with only those fields, and only for the one
person the reference names exactly (no listing, no partial matches, no 404
candidates), so a guest cannot browse the contact list. The owner also sees
`may_contact`, `cadence_minutes`, `last_interaction_at`, the per-contact
`digest` and the handles. The three mutations are refused in the plugin
outside the owner's own interactive session (a guest, a kanban worker, a cron
run) and are sent with the owner as `contact_id`, which the sidecar checks
again (403 `not_owner`). `colonymind people who|inspect|permit|cadence|merge|link|proposals`
is the same interface from the CLI. `may_contact` is raised nowhere else; a
contact's opt-out ("STOP", "don't text me", ...) only lowers it to `never`.

`colonymind_self` also carries the agent's recorded opinions (one tool, not a
separate one, so the tool schemas stay within their budget): `opinions` with an
optional `query` and `why <number>` read `GET /v1/mind/opinions` with the session's
contact, so a guest sees only everyone-audience views (an intention id is never all
digits, so `why` on any other id is still the owner's record); `withdraw` and
`reconsider <number>` with the owner's `reason` post
`POST /v1/mind/opinions/{id}/withdraw|reconsider` from the owner's own interactive
session only, never a guest, a kanban worker or a cron run (docs/OPINIONS.md). The
guard treats the tool as read-only.

`colonymind_self like|dislike` are the agent's own tastes, kept on its page about
itself (docs/SOCIAL-STATE.md, Pages): with `taste` a line is added that names the
turn it was written in, with `id` a line is removed, and with neither the page is
shown (`GET /v1/mind/pages/self`, `POST /v1/mind/pages/self/lines`). These and
`colonymind_people boundary` are the agent's own lines, not the owner's records,
so they are the one change a session that is not the owner's own may make: a
guest's session reaches the agent's own page and the page about that guest, and
no other page.

## Secrets

A secret that arrives in conversation is sealed into the sidecar's store and a reference stands in its place
(docs/SECRETS.md). The plugin's half (`secrets.py`):

- `pre_gateway_dispatch`, after the `/mind` gate: when the plugin's copy of the detector (byte for byte the
  sidecar's) finds a value in a gateway message, it posts `POST /v1/secrets/seal {text, owner | contact_id}` and
  returns `{"action": "rewrite", "text"}`; Hermes, its transcript and the model see the reference. The owner is
  named as such (the sidecar scopes the value to its owner), anyone else by their resolved contact or, when no
  contact has the handle, `handle:<platform>:<id>` (never created). The event's other words that Hermes puts in
  front of the model are sealed in place the same way: the quoted message (`reply_to_text`, in its author's
  scope) and a channel's backfill (`channel_context`, in nobody's). Without the sidecar each value is withheld; a
  value the sidecar returns as it came passes only in a span its answer marks as released.
  `secrets.capture.ingress: false` turns this off; a message with no detected value is never sent there. The
  section is validated as the sidecar validates it: a malformed one leaves every rule on.
- `tool_request` middleware: in the owner's own session a call carrying a reference has the value served
  (`POST /v1/secrets/resolve {purpose: fill, lane: owner, own_refs, action, destination}`, 32 references a
  request, more in turns) when the reference was sealed from the owner's own words in their latest message in that
  same chat (`own_refs`: what the gateway hook sealed from their text, never from a quote, a backfill, an expanded
  file or another chat); otherwise the sidecar forms an ask bound to the digest of the complete call, the
  plugin puts it to the owner in their chat directly (docs/KNOWN-GAPS.md), and the second `pre_tool_call` hook holds the call with its
  words. A value goes as text only into a send's or a write's words (`message`, `content`, `new_string` and the
  like) and into a waiting process's input as the whole of it. A terminal command or Python code never carries a
  value: it names `$COLONYMIND_SECRET_<id>` (the value) or `$COLONYMIND_SECRET_<id>_FILE` (a private 0600 file holding
  it), and the middleware hands the command to a subshell that sets both from the file (`bind_shell`) or runs the
  code with both in the interpreter's environment until it ends (`bind_python`); the command or code is never read
  or changed (a command follows the lines that set its values as it was written, so Hermes still reads its words,
  a `sudo` included; code goes as one string literal). A reference written into a command or code is refused with
  the variable to name, and a command or code naming a variable the plugin did not set for it is held. Values
  reach commands and code on the local backend only. The program that takes a file records its process and
  removes the file when it ends; `post_tool_call` removes a call's files when no program holds them, and a sweep
  removes those whose program is gone or that none took within an hour. `write_file` and `patch` in their session
  are served without an ask; a reference where no value goes (a patch's `old_string`) holds the call. In a mind
  run a worker's message to anyone but the owner has every detected value and reference replaced by
  "(a private value, withheld)".
- The body resolves an outbox message's references just before it claims and sends it (`purpose: send`, in
  batches of 32); the sidecar allows only the owner's own values, to the owner or in their dictated words, and the
  body hides any other for a contact. When any reference is not settled for now (the sidecar unreachable or without
  an answer for it, the key unavailable) the message waits for a later tick: nothing is claimed or sent.
- `colonymind_secret` (`list`, `use`, `reveal`, `release`, `forget`) works in the owner's own direct session only;
  a guest, a cron run and a worker are refused. `reveal` sends the value to the owner's own chat itself (Hermes
  would mask it in a reply) and tells the model only that it did; in a session with no chat it writes it to a
  private file (0600, removed within half an hour) and tells the model only where it is.
- A delivered turn-outbox row keeps its turn id and envelope digest and drops its payload; a rejected one
  (`last_error` "rejected: ...") keeps it in the private outbox file, the turn's only copy, for repair.

## Working-memory branches

With `mind.faculties.working_memory` on (the default), a delegated child whose parent belongs to the owner has
the text of its final message list retained as a branch. Stock Hermes fires `post_llm_call` once after the child's tool loop,
so this includes its tool results. The adapter follows the parent's recorded session lineage, requires the owner's
private session, and queues only locally established owner bindings; a hook never waits for a contact lookup or a
network call. Guests, contacts, unknown parents and kanban workers produce no branch.

Secret detection and masking run on every captured span before the private SQLite outbox is written. The local
capture ceilings are 128 KiB per span and 1 MiB per branch (implementation assumptions); material over either
ceiling is refused whole. The body delivers the sanitized branch with the parent's bound platform and sender.
Terminal refusals discard the branch payload from the outbox, with only the HTTP status logged; pending branches
older than the configured retention period are discarded when the body next runs.
The sidecar resolves that sender and checks the credential's owner platform grant again. The requester comes from
the session, never the model's tool arguments. Branch spans are owner-only ledger records, excluded from ordinary
memory claims, recall indexing and projections; retention and erasure are described in [Working memory](https://github.com/Aevonix/ColonyMind/blob/main/docs/WORKING-MEMORY.md).

The parent's next private turn gets a **Branches** context section with handles and the first line of each brief,
up to 120 characters (implementation assumption). `colonymind_consult` takes `handles` and `question`; every named
branch is read together in one tool-free model call. Results cite span handles. Stock tool availability checks hide
consults when the sidecar reports withdrawal, faculty off or the `working_memory.summary_only` benchmark arm.
The route also checks availability on every call, including calls from a session that still remembers the tool.
Setting the faculty off suppresses branch capture and the context section as well.

Only the next owner turn after a successful consult is queued as a possible outcome for the sidecar's existing
reaction reader. The adapter sends the owner's own words, not a model-supplied rating. A comparison answer that the
owner has not rated has no inferred outcome. The pending outcome binding is process-local; a process restart can
lose that label without changing retained branch material. On a sampled consult the result includes the summary
answer and an optional format for rating both answers on the next turn; see [Working memory](https://github.com/Aevonix/ColonyMind/blob/main/docs/WORKING-MEMORY.md#comparator-and-withdrawal).

Before owner-session compression, the provider queues a session branch after its existing durable checkpoint.
Delivery waits for that checkpoint to succeed; the manifest references the session's captured records and sends no
new message list. The existing provider checkpoint remains the compression barrier. When compression continues
under a new session id, the provider also lists the prior session's handles through its bounded lineage map; an
unrelated session switch inherits none. All branch-list requests share at most one second of the existing prefetch
deadline (implementation assumption).

## Memory provider

**Known limit:** a background prompt from a Hermes terminal-UI or dashboard
session starts a side agent without the session's sender. The memory provider
treats that agent as the owner's own terminal, so a session used by someone
other than the owner can receive owner-only memory through a background prompt.
Keep these terminal-UI and dashboard sessions for the owner's use only.

**Who the owner is on Hermes 0.22 and later:** on 0.22 and later Hermes has one
user. Its terminal, its terminal UI, its desktop app and every dashboard login
act as the operating-system user who runs Hermes, and ColonyMind takes that
user's own chats there as its owner: they get the owner's memory and are
recorded as the owner's turns. Give such a login to nobody else. A turn that
another agent forwards, a hosted room, an editor client, a task or a scheduled
run under its own label, and a message delivered into the profile's Bot Chat
that names another author are not the owner's, although Hermes runs them under
the same user. But on 0.22 and later Hermes names the author of a turn only when
the sender gives one. A message delivered into the profile's own Bot Chat
without an author runs as the owner's turn: it gets the owner's memory, it is
recorded as the owner's words, and its answer goes back to the sender (for a
scheduled report a receipt goes back and the answer stays in the chat). That is
so for a report that a scheduled job places there, whether a job of this profile
or of another profile on the same machine, for `hermes peer dm` and
`hermes peer run`, and for any client that holds the profile's API-server key;
and a sender can also name the owner as the author.
So give the API-server key to nobody you would not give the owner's own login,
as with a dashboard login, and do not use ColonyMind's memory in a profile that
takes other people's words into its Bot Chat.

When supplied to the provider's per-session turn binding, the host's turn id is sent as `context.turn_id` only for an owner-private prefetch; contacts, guests and turns without an id keep the existing body.

`prefetch` assembles context for the turn's participant. A guest request sets
`audience: viewer`, and the sidecar returns a guest only that contact's scoped
sections, never an owner-only one. A kanban worker's prefetch names its task
(`context.metadata.kanban_task`): its task body already carries its lessons, so
the sidecar adds no turn lesson and logs no lesson use for it. A scheduled job's
prefetch (Hermes builds its agent with platform `cron`) says so
(`context.metadata.run: cron`): a job's prompt is not the owner's turn, so the
sidecar leaves the owner's status block out of it. Hermes waits
for the whole prefetch, so each one has one deadline from its start (Hermes'
external prefetch bound less the provider's 0.5 s margin): the participant's
resolution, the assembly and the Current Time refresh all end by it, each
bounded in wall-clock time as a whole (an HTTP client's timeout bounds each
read, so an answer arriving in slow chunks would outlast it): past the
deadline the provider stops waiting and closes that request's connection. The
assembly request names what is left (`wait_ms`), the sidecar's recall spends
only that (docs/RECALL-HYBRID.md, Recall latency), and the Current Time refresh
gets what the assembly leaves (every refresh while sections are held back
included); with nothing left the turn carries the provider's own clock line. The scoping
fails closed: anyone the sidecar cannot show to be the owner, a caller without
the key included, gets the contact-scoped set. Recall includes this session's
own earlier turns: Hermes compacts sessions in place, rebuilds agents after an
idle eviction or a restart and compacts on detached agents, so no provider
instance knows whether the model still reads them verbatim. The provider's
direct tools are offered on the owner's own lane only: a guest session or a
channel with no sender binding gets none of them, and a call that still
arrives, like `colonymind_memory_search` for a turn with no resolved
participant, is answered once with `{"unavailable": true, "retry": false,
"reason": ...}` rather than an error the model retries. `sync_turn` is active
only when the general plugin is not enabled; otherwise the outbox owns capture.
`on_pre_compress` writes a checkpoint through the same outbox before Hermes
compresses a session, bound to the turn's exact participant. When the sidecar
cannot answer the sender's lookup right then (unreachable, out of time, its
breaker open, a server error), the checkpoint keeps the participant that sender
resolved to earlier in the same session (`participant: kept` in the provider's
last checkpoint); a sidecar that says the sender has no participant, another
session or another sender never falls back. The sidecar's answer is about the
sender: once it says the sender has no participant, or now resolves it to
another one, no session (a compressed copy included) keeps the old participant
for a later outage.

Hermes asks `is_available` on its hot path and drops a provider that says no
for the session, so the check reads configuration and files only, never the
network: it fails without a sidecar URL (`plugins.colonymind.sidecar_url`) or
without a readable, non-empty key file (or `COLONYMIND_API_KEY`).
`unavailable_reason` names which, with the fix, and
Hermes prints it. A sidecar that is down is not unavailability: each request
reports it and the provider stays active, so captures continue through an
outage. `backup_paths` names the instance directory, so `hermes backup`
carries the stores, the key and the configuration with the profile.

`recall_status` feeds Hermes' recall indicator: after a prefetch whose memory
section cites N quotations, Hermes shows "recalled N" as a status line by default.
Only the owner's private turns (a direct chat or the terminal) produce notices;
a contact's turn or a group never does. The origin status line stays outside the
model's context; Hermes may mirror routed notices into the destination chat's history.

Configure notices in Hermes' config, under `plugins.colonymind.recall_notice`:

```yaml
plugins:
  colonymind:
    recall_notice:
      destination: origin
      details: false
```

`destination` defaults to `origin`, which keeps Hermes' existing compact status
line. `home` sends to the turn platform's configured home chat: the provider passes
the bare platform to Hermes and lets Hermes resolve it. CLI and TUI turns fall back
to `origin`. `off` disables notices, and `<platform>:<chat_id>` sends to an explicit
chat. An unknown destination falls back to `origin` and is logged once.

`details` defaults to `false`. Set it to `true` for short, redacted memory previews
with dates in routed notices. Other people's memories contribute only an aggregate
count, never preview text. The origin status line stays unchanged. Routed sends
start only when Hermes calls `recall_status`, never during prefetch, and run in the
background so recall does not wait for a message send. Each process allows one
outstanding send. Notices arriving while it is busy are skipped and logged, without
a queue or retry. A send lasting over 10 seconds logs a warning and keeps its slot
until it returns; the warning does not cancel the transport. Provider failure warnings are
rate-limited and omit message text, tool results and destination identifiers.

Hermes appends each turn's recalled context to that turn's user message and
replays it byte for byte on every later turn (the `api_content` sidecar). The
provider keeps a fingerprint of each context section it showed, per session, and
a section unchanged since the session's last shown turn is one line in the new
turn's context: `unchanged since your last turn: <section>`. Only the new turn
changes; earlier turns are never touched, so the cached prompt prefix holds.
Current Time is always sent. A prefetch counts as shown only once its turn
completed (`sync_turn`, called whoever writes the turn), only when Hermes waited
for it (its 8 s external prefetch bound) and only when it went inline whole
(under Hermes' spill cap). The fingerprints reset on every session switch
(including `/undo`), before compression (`on_pre_compress`) and on restart; the
gateway's detached compaction evicts the live agent, whose provider starts
empty.

The recall-exit counter observes memory context actually submitted to the model.
In the instance's `colonymind.yaml`, `context.recall_alarm_turns` (default 2)
sets the consecutive owner-turn threshold for `RECALL_ALARM`.
`context.recall_alarm_target` optionally names a stock Hermes send target;
only a non-empty string enables delivery, with missing, blank or invalid values
keeping the existing log-only behavior. Both settings are read at plugin startup.
Body passes drain observations and queue one pending notice. Only the Body loop
thread takes that notice after `run_once()` and sends it outside the counter lock.
The notice text is
`Memory context was missing on N consecutive owner turns - exit codes: CODES.`
and goes through `body.send_message`, using Hermes' stock send path. Codes are sorted
and comma-separated; the text is ASCII with no media tags or turn/session ids.
`flush()`, `tick()` and caller-thread `run_once()` never send recall alarms.
With the Body thread disabled, the alarm is log-only. A newer streak replaces
an unsent notice in the single pending slot. There is at most one attempt per
streak, re-armed by the next `ok`. An `ok` before the loop takes the notice drops
it; one already taken is still sent. A failed, raising
or uncertain sender never triggers a retry. `RECALL_ALARM_SEND` records only
`result=sent`, `result=failed` or `result=uncertain`, without target or text.
Response hooks enqueue only and never wait for delivery.

The context never outgrows Hermes' inline cap (`hooks.output_spill.max_chars`,
10,000 by default), which would replace the whole of it with a head, a tail
and a file path. The provider names its inline allowance (`inline_chars`, the
cap less 256 for the clock it adds) and the sections it will send as one line
(`shown`, their fingerprints); the sidecar's per-turn selection fits the
context to that allowance, and the turn's resident blocks are fitted to the
same budget: a hard limit per block, fresh state admitted first, whole lines kept
and a marker for what was held back, and a `Held back this turn` section
naming any block that did not fit (`sidecar/colonymind/memory/blocks.py`,
architecture 4.1). Against a sidecar that fits nothing (1.10) the provider
holds whole sections back itself, fresh state last, and names each one.
Current sidecars always fit context. `context.budget_chars` caps the full assembled
sections even when the host offers more room or will collapse unchanged sections.
`context.recall_items` caps recalled records (owner-adjustable default 20), including after a
verified reranker cutoff. This matches the independent 20-candidate rerank bound;
unverified or missed reranking keeps its five-record fallback. `COLONYMIND_CONTEXT_HOST_WAIT_MS` caps the whole assembly
(default 7,500 ms); `wait_ms` can shorten it. An expired assembly returns no sections. Synchronous producers use a
dedicated 16-worker pool. Healthy concurrent reads wait for a free worker within the turn deadline; work is never
queued in the executor. When abandoned reads fill the pool, a producer is skipped and counted in
`budget.sync_pool_skipped`. Required reads fail closed; only reads whose absence hides evidence use empty defaults.
Worker reads open short-lived read-only SQLite connections; both statement entry and SQL progress check the deadline.
An abandoned read holds its slot until it finishes. Read failures increment `budget.sync_read_failed` and log one
warning per turn. The final deadline check commits to the response, then hands its captured writes to a separate
event-loop task outside the assembly time box. A blocking write can delay later loop work, but cannot make this
assembly discard the response it already accepted. Recall records run only for successful recall kept in that
response. Late read results and discarded producers' writes are dropped.
Remove the retired `context.blocks` key from configuration.

## Tests

`tests/hermes_adapter` runs against stock Hermes installed in the same
environment plus a fake sidecar; see `tests/hermes_adapter/conftest.py`.


## Durable hands plugin

The separate `plugins/colonymind-hands` source plugin implements controlled
background delegation. It wraps the native delegation entry point and both
async dispatch functions without changing Hermes source files. Load it through
its source manifest. Distribution entry-point wiring remains an integrator task.

Configure `hands.platforms` explicitly after verifying detached completion
routing, then provide provider contracts and an optional role resolver. The
empty default allow-list refuses delegation. Unsupported sessions and full slots
return a refusal without running the task inline. All hand tools retain local
budgets and authority checks during a controller outage. Ordinary owner tools
keep their existing policy.

The host may register a notice sink; the plugin never chooses a recipient.
Hands return their work to the assistant's main conversation. Monitoring goes
through the sink, with separate routine and actionable limits. See the
[Hands plugin section](https://github.com/Aevonix/ColonyMind/blob/main/docs/HANDS.md#plugin) for dispatch, durability, controls,
accepted shell limitations and the integration checks.
