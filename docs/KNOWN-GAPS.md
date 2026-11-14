# Known gaps and retired scaffolding

This is a source inventory, not a production health report. Phase 1 establishes
the first supported ColonyMind baseline and is still being validated. Current
behavior and limits live in the README and capability guides. Enabling a
profile still requires checking its actual effects.

Older ColonyMind releases, aliases and migration paths carry no public support
promise. Retained modules are listed here so cleanup can account for their
actual callers and data, not to require compatibility with every old release.

Known current limits include complete erasure of unlinked historical data and
host transcripts, cross-database recovery, empirical retrieval quality beyond
the tested cases, and deployment-specific hardware coverage. Native skill
evaluation needs an independently supplied task oracle; it does not establish
general self-improvement simply by recording a successful review. Working-memory
branches keep captured worker spans as owner-scoped ledger records: source
erasure removes them from branch renders, so branch storage does not add an
untracked transcript copy. This does not erase a host's own transcript files.
See [Working memory](WORKING-MEMORY.md).

## Reply waits without a producer

With registration and the follow-up plan routes removed, nothing creates a temporal follow-up (reply
wait) any more. Existing waits are still read, changed, reconciled against transport receipts and used
by the mind's reply-wait duty and reply forecasts until they resolve. The contact reply-timing learner
no longer depends on them: it reads the comms ledger (see below).

## Contacts' reply timing sees only logged conversation

`contact_reply_timing` learns from the comms ledger's conversation rows. The agent's own tool sends to
contacts (a message tool call during a turn, the mind's `mind:` outreach rows) are not conversation rows,
so the learner sees only the conversations in which it answered a contact's message. Logging those sends
is not part of this build. A transport outage reads as no reply: a send whose reply never reached the
ledger is labelled 0 after its 24-hour horizon. Volume is low: on the reference instance, 90 days held two
labelled sends, and the learner needs 30 training and 30 reserved-week labels before a revision can be
qualified, so it serves the fixed 0.5 base until then.

## Nightly budget admission

A nightly call is admitted on the larger of its output cap and the biggest call so far; its own prompt
is not reserved in advance. A night can therefore end above its share by up to one prompt. The single
length retry is admitted with the first attempt's reported prompt included, so the retry itself cannot
push the night over.

## Closed in the job-watch build

Learner polling and coverage reconciliation no longer hold up the event loop. Keyword ranking can use
its temporary FTS5 tables again. Failed context reads and dropped writes now warn and increment the
owner's `health_counters`; a loop-lag recorder records stalls. Backups create private directories and
files, preserve an existing destination directory's mode and keep owner execute bits.
A cut-off nightly answer gets one budgeted retry, and unfinished stages make the night
report `incomplete`. These changes expose failures; they do not guarantee a complete context or night.

The work list again includes the agent's own Hermes background processes. Owner check-ins set from
conversation now reach due outreach under its existing limits. Ignored sends count only inside the
seven-day novelty window, ending the month-long outreach lockout. Documented idea-engagement fields
are accepted and validated, though `value`, `anchor_kind` and `metadata` are not retained.

## Partially wired (works, with a missing half)

- **Job watch**: optional and scheduled independently; no timer is installed automatically. Without
  `notice_command`, findings are recorded but no notices are queued or logged. Delivery is at least once,
  so a crash after a successful send can repeat a notice. Without `/proc`, CPU is paired with a digest of checkpoint output when
  present, a one-shot cron claim heartbeat, or a kanban run heartbeat. On macOS, Hermes background
  processes are never judged stuck: Hermes keeps their output in memory and its checkpoint has no
  output field. Their completion failures and repeat checks still work. Over-time checks remain
  independent of stuck detection, but current background-process records supply no runtime limits.
  Kanban runs use their per-run heartbeat alongside CPU time.
  Recurring cron runs, overlapping cron attempts, and kanban runs without a heartbeat also remain
  unjudged there: CPU alone cannot distinguish a hang from a network wait. Cron claim heartbeats and
  CPU from unrelated work in a shared scheduler can mask a hang. Detached tmux shells at their prompt
  are excluded even with background children, using the terminal foreground process group from `ps`;
  without that measurement, only childless shells are excluded. Explicit pane commands (including shell
  scripts) remain watched. Other interactive programs waiting for input can still look stuck. Linux write counters count all writes,
  not just useful output. Cron and kanban records do not expose turn consumption, so watch cannot prove
  their turn overruns. Only declared runtime limits are checked. Hermes completion receipts are retained
  for seven days or 64 files, and replay cannot reconstruct vanished receipts or old CPU/output samples.
  Unreadable sources do not close findings; sampling gaps restart idle/download intervals. The work
  list's process reader is bounded to 150 ms and reports unavailable coverage when it cannot read.
  Health counters reset with the process; daily summaries start from the first successful baseline read
  and report only dropped context writes and incomplete mind tasks.
  See [Job watch](JOB-WATCH.md) for the exact checks and delivery behavior.
- **Memory consolidation in the benchmark**: the memory and self families cross
  one night before the probe (amended 2026-09-24), so `full-consolidation` and
  `full-self_narrative` can differ from `full`; no pilot has measured it yet. The
  per-contact digest stage writes only into a contact store with the people
  milestone's digest columns (`set_digest`), so on a store without them the
  consolidation arm differs by the narrative, contradictions and episode
  summaries alone. The `semantic_recall` arm differs from `full` only in a plan
  given an embedding endpoint (`paired plan --embedding-config`).
- **Lessons (M9)**: which lesson reaches a task body or an owner turn is decided
  by meaning (`P/mind/lesson_match.py`): the cosine of the lesson's and the
  work's embeddings against a bound measured for the running embedding on 80
  synthetic pairs, which may not hold on live lessons (each use records how it
  matched, so the campaign report's lesson diagnostics can split the use rate).
  On ColonyMind's default embedder (Qwen3-Embedding-0.6B) no bound keeps the
  precision floor on that set. A shipped bound applies only to the embedding
  identity it was measured on (through an endpoint, at a declared revision),
  so a deployment on another one (a local provider, another served model
  name, no declared revision) is unmeasured and gets the default 0.72, at
  which neither shipped measurement relates any of the 80 pairs. Either way
  such a deployment gets few or no lessons by meaning (a task's own failure
  class still) until it records a measurement of its own
  (`COLONYMIND_LESSON_MATCH_BOUNDS`, at a floor it chooses). A lesson
  admitted while no embedder answers has no vector until the night's lesson
  pass. The owner's turn pays one more embed call when recall's vector lane
  did not embed the same text (the lane is off, or it searched the owner's
  words with the mind's broadcast concerns added); when that lane's embedding
  timed out, the turn gets no lesson by meaning rather than a second attempt.
  A `result_field` check passes on the worker's own summary, so it verifies
  no lesson (the other faculties still read it as `check`). Lesson uses are
  joins over intention rows and use notes, so the tally covers the 90-day
  audit retention and no more. An owner verdict in conversation counts only
  when the night's model quotes the owner's exact words; a verdict it does
  not recognise scores and teaches nothing.
- **Owner outreach (M11)**: the owner's replies are read by a lexicon
  (`P/mind/reactions.py`) with two appraisal nets (the owner's opt-out and a
  dismissal after a send); a reply worded outside it and not caught by the
  nets is linked only as engagement or not at all, so recall on held-out
  phrasings is unmeasured until the family's pilot, which reports it per class
  (a follow-up adds one tool-less classification call per linked reply if
  positive-reply recall is under 80%). A position link needs a short or
  referring reply, the owner's first turn after the outreach, not
  mid-conversation, with no request of its own, so a reply that names neither
  the topic nor the item and misses any of these links nothing and the
  outreach is scored by silence. Relevance is lexical (shared terms with the
  owner's interests and goals, and a sentence of the owner's naming the
  topic), and so are the null-report and repeat checks (a report worded as a
  finding that says nothing new in new words passes them). "Not now" teaches a per-hour mark, not a schedule:
  reply-timing learning can now hold discretionary outreach for at most two hours
  through a configured kind D learner with a live comparator and automatic rollback;
  presence features and channel choice are not built. A finding whose text
  carries a price trips the authority floor's money pattern and becomes an ask
  instead of a message; the family's items carry no currency. The faculty is
  measured by the `mind-outreach-1` family (section 5 of
  `docs/proto-agi/families/mind-outreach-1.md`) and ships on; the README's
  evidence section gives the result.
- **Skills (M9)**: off by default. Loads are counted only for ColonyMind's own
  `colonymind-*` skills, and a process whose Hermes lacks
  `clear_skills_system_prompt_cache` lists a new skill only after a restart.
- **Recall reference numbers**: the earlier three-arm reference results are
  retired. Publishing current results needs a measured run through
  `collect_sources`/`select_memory` with declared extraction, embedding and reranker endpoints.
- **Approvals**: the owner answers an ask from their own message (`docs/MIND.md`, Asks). What stays open, and what
  the approvals design asks for that is not built yet:
  - Whoever can read the instance key can forge an owner turn to `/v1/mind/answers`, as to any turn route, and so
    approve an ask. Not every approval reaches the owner's chat: a reaction, and a message read from a captured
    turn, get one line there saying what they did (`ask_ack`), but what a text answer on `/v1/mind/answers` did goes
    back to its caller only, for the agent to acknowledge in its reply. So a forged text answer can approve an ask
    with nothing in the owner's chat. A tool call that reads the key or names an answer route is noted
    (`answer_forgery_attempt`, told to the owner at most once a day). In the owner's own lane the call runs; in a
    non-owner session and a mind run it is refused. That rule reads how a call spells a name, so it is no boundary
    where a shell runs: the owner's own lane, and a non-owner lane the operator opted in
    (`guests.executor_platforms`). `colonymind doctor` names the remedy for both (`owner-lane-tools`,
    `guest-lane-tools`): run those lanes' terminal, code and file tools where the key cannot be read.
  - A non-owner session may call only the guest tools; a tool an operator adds with `guests.tools` is trusted as
    named, except that every local file named in its arguments must be one the sender sent (an MCP tool that can
    run code, or reads a file it is not given by name, is the operator's choice to add), and a platform in
    `guests.executor_platforms` keeps every tool, where only the spelling rule stands between a shell and the key.
  - The guard sees tool calls, not the model's final reply: a `MEDIA:` path in a guest's reply is delivered by Hermes'
    own media rules (its denylist; `HERMES_MEDIA_DELIVERY_STRICT=true` in Hermes' `.env` limits delivery to its
    caches and fresh files, which Hermes advises for public-facing gateways).
  - A session nothing tells about (not seen by the plugin, no parent to follow in Hermes' store, no gateway binding
    for the turn) is left alone, as before: refusing it would stand in the owner's way when a lookup fails, and no
    contact reaches the delegation or scheduling that makes such a session without an opt-in.
  - An owner sender known only by the sidecar's contact resolution (not by a handle in identity.yaml) reads as a
    non-owner while the sidecar cannot answer, and gets only the guest tools then; listing the owner's handles in
    identity.yaml removes that dependency.
  - No floor grace: a yes to a floor ask starts the work at once, so an undo cancels only what has not started. The
    asks report's grace cancels stay at 0.
  - A lapsed ask cannot be revived: an answer after the lapse approves nothing, a quoted reply on the ask's own
    message included. The report's revivals stay at 0.
  - The mind asks less only through "always" and the offers to stop asking (`docs/MIND.md`, Standing allows): a
    wish, a check-in matched to exactly one person and a people change the owner states in their own words still
    ask.
  - The guard does not refuse the model's own reaction or quoted reply on an ask's message (stock `send_message`
    can react on some adapters). An event a channel marks as the agent's own account's is never read, but only the
    WhatsApp bridge's owner echo is marked, and no stock adapter sets a forwarded mark, so a forwarded message reads
    as typed.
  - A sealed value's use (`docs/SECRETS.md`) is asked by the plugin directly, in its earlier words with the ask's
    code. That message is not in the chat ledger, so a bare yes right after it binds nothing, and a code the owner
    types only shows the ask again as its own message, which the owner then answers.
  - A yes said on a voice call answers no ask; the ask waits in the chat.
  - The invented code lines the plugin removes from replies are counted only in its own log
    (`invented_ask_reference`), which never reaches the sidecar. The asks report's line "Invented code lines
    removed" reads 0 from a running sidecar all the same: that 0 is not a measurement.
  - No poll is sent where an adapter could send one; reactions and quoted replies are read.
- **The MCP endpoint** (`docs/MCP.md`): unmeasured. No host but Hermes has run a
  measured task against ColonyMind yet, so what its reads and notes add to
  another host's agent has no number. It serves plain HTTP with a bearer
  credential and no TLS of its own, answers with JSON only (no server-sent
  stream, no resources or prompts), and a note is kept only as the credential's
  session evidence: nothing reads it into a claim, by design.

## Retained code outside the baseline

- The incompatible manual `plugins/hermes-context/` compressor has been removed.
  Use the native context engine with the current general and memory-provider
  adapters.
- Goal records, saved DAG history, GET/PATCH APIs, context and condition updates
  remain readable and editable. The duplicate planner, decomposition, queue
  dispatch, conversation synthesis and goal-creation endpoint are removed.
  Hermes owns executable tasks. Reported goal completion does not verify an
  outcome or stop a worker; it preserves its first completion time and commits
  the record and transition together.
- The unused tier learner and quality-outcome API have been removed. Named
  function routing and configured fallback remain; explicit tier selection uses
  deterministic thresholds. Old `router_self_learning.db` files are neither read
  nor written and can be discarded.
- SQLite holds canonical source memory. The
  Neo4j memory graph, the SQLite world model, the belief engine, the chain and
  the continuous learner were removed in M8; `colonymind upgrade` moves their
  local state files into the upgrade backup. Changing databases does not fix
  memory admission quality or recover missing provenance.
- Graph consumers outside the deleted packages await the M10 audit. Those that
  only optionally read the graph lost that branch (the insight validator's
  data-age check, the relationship scorer's and cognition components' type
  hints); the research pipeline's graph stage (`GraphGatherer`, its source type
  and its web-versus-graph contradiction rule) is deleted. Those that cannot
  work without a Neo4j driver (`SignalCollector`, which M6 deleted with
  `mind_model`) are no longer constructed by the server. The cognition pipeline
  (the MetaLearner, the CPI and the strategy adjuster) was deleted in M9. The
  `extraction` extra is deleted; the research pipeline, the synthesis and
  insight routes, the briefings, the sessions and the agent registry with its
  node-certificate signer went in the prune (below).
- The prune removed what the Hermes body already does and what nothing used:
  the agent SDK and registry, briefings, channels, connectors, the legacy goal
  records, the MCP server and its harness setup, observations, patterns,
  proposals, repo mirrors, research and search, sessions, the event bus and its
  models, the situation spine, the intelligence components other than the
  preference learner, and the shared facts. `colonymind upgrade` moves their
  stores into the upgrade backup (never deletes them). The competence
  store still supports the action journal at `GET /v1/host/self/journal`.
  The self-model and benchmark read routes have been removed.
- The execution registry, execution and task-run forecasts, diagnostics, native-learning pack,
  initiative assignment leftovers, tool observations, commitment local work and drafts, task-instruction
  and transport-media routes, and hands park and triage stubs are removed, not unfinished features.
  Removed routes return 404 or 405. Upgrade backs up the retired execution tables and removes unread
  hands settings. The executions read still shows selected live work; the journal and public document
  capture route remain.

## Deliberate no-builds (division of responsibility with the host agent)

ColonyMind is the cognitive substrate; the host agent framework (e.g. Hermes)
owns sessions, tool execution, message transport, and cron. These stay
unbuilt HERE by design:

- **Email/desktop/browser job handlers**: outbound messaging goes through
  the host gateway (delivery bridge); ColonyMind never sends email itself. The
  desktop/browser packages were scaffolding for host-side capabilities and
  the dead EmailHandler has been removed. `JobType.DESKTOP`/`BROWSER`
  remain enum values with no handler.
- **ScheduleAdapter**: removed. Its contracts were
  unimplementable (the MetaLearner, deleted in M9, had no pattern API; the
  AutonomyScheduler is interval-based, not a cron store) and mutating host
  cron jobs would cross into the host framework's domain.
- **Initiative execution requires Hermes**: registered evidence reviews use existing native task bindings. Follow-ups use the
  native preparation and delivery path. Unsupported work remains visible as
  proposals; proposal text is never an execution grant. The duplicate no-host
  executor, its startup flag and status endpoint are removed; the upgrade
  reconciles its queue and migrates the historical skill counter.

## Removed during the 1.0 consolidation

The following modules had no runtime imports, registered entry points or
configured loaders in this repository. They are retained in Git history rather
than offered as unfinished features:

- Raft consensus and its isolated unit suite, plus the unregistered chain key,
  sentinel and administrative CLIs. (The rest of the chain followed in M8.)
- The federation skill marketplace and its unused protocol, plus the unused
  skill schema-version helper. Hermes owns the supported native skill review
  path; executable work uses the supported native task bindings.
- The unused structured-world importer and email-header contact importer.
  Supported contact import paths remain.

The unused `gate/pending_dispatch.py` re-export, default cloud subtask handler,
inert operations scripts, webhook examples and old patch inventory runner are
removed. Explicitly registered custom workers remain supported; Hermes runtime
qualification uses the packaged patch set.

## Known mechanisms (documented so the log noise is interpretable)

- **"Unclosed client session" (aiohttp) after tick-budget cancellations**:
  when a tick exceeds `COLONYMIND_TICK_BUDGET_SECS` the whole-tick `wait_for`
  cancels whatever await is in flight; a cancellation landing inside an
  aiohttp request can interrupt the session unwind and the GC later logs the
  unclosed session. Mitigated by strongly referenced background
  tasks; the world-LLM extraction and the per-call graph driver that also
  produced it are gone (M8). Residual noise right after a budget-exceeded tick
  is expected and harmless.

## Settlement semantics (by design, documented here so nobody "fixes" it)

- Workspace concerns raised from **commitments** settle durably on resolve
  (the source closes). Concerns raised from **anomalies / benchmark
  regressions** have no settler: resolving them suppresses the dedup key for
  `COLONYMIND_WORKSPACE_RESOLVED_TTL_HOURS` (default 24h), after which a source
  that is STILL firing legitimately returns. That re-raise is intentional:
  a day-old still-live anomaly deserves attention again.
