# ColonyMind documentation

The guides, by task. The [project README](../README.md) says what ColonyMind is and how to install it in three
commands; the [changelog](../CHANGELOG.md) says what each release changed.

## Start here

- [Install and update](INSTALL.md): requirements, the three commands, what `colonymind init` changes,
  [the container](INSTALL.md#container),
  [coming from another memory provider](INSTALL.md#coming-from-another-memory-provider),
  [configuration](INSTALL.md#configure), upgrading and uninstalling.
- [Compatibility](COMPATIBILITY.md): each release against the Hermes and Python versions it supports.
- [The Hermes adapter](HERMES-ADAPTER.md): how ColonyMind attaches to stock Hermes, the seams it uses and what
  each hook does.
- [Troubleshooting](TROUBLESHOOTING.md): common symptoms, how to tell their causes apart, and the fixes.
- [Importing Hermes history](HERMES-HISTORY-IMPORT.md): bring reviewed quotations from an existing Hermes
  store into memory.
- [Architecture and ownership](ARCHITECTURE.md): what Hermes owns, what ColonyMind owns, and how they meet.
- [Known gaps](KNOWN-GAPS.md): what is partly wired, retired or deliberately not built.

## Using it

- [The mind](MIND.md): commitments captured from conversation ([the loop](MIND.md#the-loop)), drives,
  concerns, authority and [the floor](MIND.md#the-floor),
  [standing grants](MIND.md#standing-grants) and [allows](MIND.md#standing-allows), [asks](MIND.md#asks),
  [outreach to the owner](MIND.md#outreach-to-the-owner),
  [the daily digest](MIND.md#the-daily-digest), [the off switch](MIND.md#the-off-switch),
  [reports](MIND.md#reports) and [the CLI](MIND.md#the-cli).
- [Instinct](DECISIONS.md): the mind's quick typed decisions, local tiers measured in shadow and promoted only on
  evidence.
- [Hermes gates](HERMES-GATES.md): optional gates that skip a cron run with nothing new, triage inbound messages and
  screen tool results, measured in shadow before they act.
- [Reminders from recalled deadlines](SOURCE-REMINDERS.md): a reminder scheduled from a remembered claim.
- [People](RELATIONSHIPS.md): one person across every channel, permissions to reach out, check-in cadence.
- [Contacts and appraisals](SOCIAL-STATE.md): identity links, appraisals and what the agent knows of each
  person's current state.
- [Opinions](OPINIONS.md): the agent's reasoned, revisable views and how they change.
- [Nightly consolidation](CONSOLIDATION.md): the self-narrative, lessons, contradictions, digests and episode
  summaries.
- [Feeds](FEEDS.md): topic briefings built from a spec file.
- [Checks](CHECKS.md): mail, repositories and other sources watched as scheduled jobs that wake the model only for
  something new.
- [Agent mailbox](agent-mailbox.md): receive-only identity, inbox/thread contract and
  [provider provisioning proposal](agent-mailbox-provisioning.md).
- [The public API](API.md): the routes a client can rely on, and each release's `openapi.json`.

## Memory

- [Memory quality](MEMORY-QUALITY.md): what is kept, the one assertion extractor, and what makes a memory
  useful.
- [Hybrid recall and calibrated abstention](RECALL-HYBRID.md): keyword and semantic search, the reranker, the
  calibrated cutoff, per-turn context selection and recall latency.
- [Semantic recall](SOURCE-SEMANTIC-RECALL.md): recall by meaning from the original messages.
- [Voice recall](VOICE-RECALL.md): the capped, low-latency recall route for spoken turns, with speculative
  search on partial transcripts.
- [Voice affect line](VOICE-AFFECT.md): one short, read-only line of the speaker's and the assistant's current state.
- [Context card](CONTEXT-CARD.md): the owner's versioned daily summary that a voice client fetches and pre-warms.
- [Voice write-back dedup](VOICE-WRITEBACK-DEDUP.md): voice rows copied into a chat session are not captured twice.
- [Embedding generations](EMBEDDING-GENERATIONS.md): model swaps and rebuilding the vector index.
- [Claims](SOURCE-CLAIMS.md): dated assertions, corrections and disagreements drawn from what people said, the
  extraction [dry run](SOURCE-CLAIMS.md#dry-run) and [re-extraction](SOURCE-CLAIMS.md#re-extraction) of older
  history.
- [Corrections and the memory inventory](SOURCE-ANNOTATIONS.md): attributed corrections added beside the original
  evidence, and [what memory holds about a person](SOURCE-ANNOTATIONS.md#the-memory-inventory), with each claim's
  history.
- [Backup and recovery](SOURCE-MEMORY-RECOVERY.md): recovering source memory from a backup without undoing an
  erasure.
- [The memory twin](TWIN.md): a live, read-only replica on a second machine.
- [The atlas](ATLAS.md): an offline, owner-only site of what people said, cited.
- [The owner console](CONSOLE.md): read-only pages on this machine for the mind, the reports and memory, with the
  off switch.
- [A correction across a model swap](MODEL-SWAP-DEMO.md): a small end-to-end demonstration.

## Media

- [Images](SOURCE-IMAGES.md), [audio](SOURCE-AUDIO.md), [documents](SOURCE-DOCUMENTS.md) and
  [video](SOURCE-VIDEOS.md): originals kept with provenance, descriptions and transcripts recalled, and the
  original reopened on request.
- [Observed conversations](SOURCE-CONVERSATIONS.md): conversations a wearable or room microphone heard around the
  owner, kept word for word with their speaker labels; only the owner's own words become the owner's claims.

## Security and privacy

- [Credentials](CREDENTIALS.md): who each API credential may write as, and on which platforms a turn can be
  the owner's.
- [Secrets](SECRETS.md): secrets that arrive in conversation are sealed out of memory and used by reference.
- [The owner gate](OWNER-GATE.md): what reaches the owner unprompted, and the audit of each verdict.
- [Owner approvals](APPROVALS.md): answering held calls in the owner's chat and recovering an unbound answer.
- [Source erasure](RELATIONSHIPS.md#source-erasure): forgetting a source tombstones its recall evidence while retaining historical data.

## Models and execution

- [Grounded Ideas](grounded-ideas.md): source-validated offers, the presentation the app shows and the owner-started Hands Build queue.

- [Model roles](FUNCTION-ROUTING.md): named functions, their models and fallbacks, and
  [what the model calls cost by lane](FUNCTION-ROUTING.md#usage-model-calls-by-lane).
- [Shared inference capacity](INFERENCE-POOL.md): one admission boundary for Hermes and ColonyMind model calls.
- [Model qualification](MODEL-QUALIFICATION.md) and [portable behavioral packs](QUALIFICATION-PACKS.md):
  recording the quality and latency of the model behind each role.
- [Runtime logs](RUNTIME-LOGGING.md): where the sidecar logs and how the files rotate.
- [Job watch](JOB-WATCH.md): optional independent checks for stuck jobs and broken dependencies, findings
  and opening/recovery notices; [timer installation](INSTALL.md#optional-job-watch-timer).
- [MCP](MCP.md): a read-first endpoint for other agents, bound to a credential of their own.

## Measurement

- [The paired benchmark](PAIRED-AGENT-BENCHMARK.md): stock Hermes against Hermes with ColonyMind on the same
  scenarios, one faculty at a time.
- [Frozen workflows](FROZEN-WORKFLOWS.md): work that has to survive session boundaries, corrections and
  restarts.
- [Proto-AGI evaluations](proto-agi/PROTO-AGI-EVALS.md) and the benchmark families:
  [initiative](proto-agi/families/mind-initiative-1.md), [drives](proto-agi/families/mind-drives-1.md),
  [people](proto-agi/families/mind-people-1.md), [affect](proto-agi/families/mind-affect-1.md),
  [opinions](proto-agi/families/mind-opinions-1.md), [memory](proto-agi/families/mind-memory-1.md),
  [the harder memory family](proto-agi/families/mind-memory-2.md),
  [improvement](proto-agi/families/mind-improve-1.md), [outreach](proto-agi/families/mind-outreach-1.md),
  [recurring failures](proto-agi/families/mind-recurring-1.md).
- [Public memory benchmark](PUBLIC-MEMORY-BENCHMARK.md): memory measured on public datasets at matched budgets,
  and the published results.
- [Public benchmark results](PUBLIC-BENCHMARK-RESULTS.md): exporting qualification runs for publication.
- [Release qualification](RELEASE-QUALIFICATION.md): how the published packages are checked.

## Design and history

- [The Proto-AGI architecture](proto-agi/PROTO-AGI-ARCHITECTURE.md) and its
  [build plan](proto-agi/PROTO-AGI-BUILD-PLAN.md): the design of record and its milestones.
- [Plasticity](PLASTICITY.md): learning in weights, governed like memory; the core's contract as built, not yet
  wired.
- [Plasticity learners](PLASTICITY-LEARNERS.md): service health, reply timing, and the query adapter.
- [Hands](HANDS.md): durable delegated work (tasks, attempts, evidence and obligations); the seam as built, not
  yet wired.
- [Persistent Hands](HANDS-DEFINITIONS.md): named definitions, task snapshots, delegation, scoped credentials and change events.
- [Design notes and historical studies](internal/README.md): how individual parts were built and measured.
- [Artwork](branding/README.md): the logo files used in this repository.
