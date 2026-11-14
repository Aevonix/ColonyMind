# Design notes and historical studies

These pages record how parts of ColonyMind were built and measured: design notes for single increments, the
build records of the qualification packs, and the older agent studies. They are kept for the record and for
contributors. They are not guides for running ColonyMind, and some describe behavior a later release changed or
removed; the [documentation index](../README.md) lists the current guides.

## Design notes

- [Connecting Hermes](HARNESS_INTEGRATION.md): the integration boundaries of the two Hermes registrations.
- [Session standing and unknown authority](guard-standing.md): creation evidence, inherited authority and
  fail-closed guard behavior.
- [Source lineage of the contacts' stores](TOM-SOURCE-LINEAGE.md): how the comms log and contact affect stay
  linked to their sources, and what forgetting removes.
- [Owner preferences](WORKING-PERSPECTIVE.md): communication and initiative preferences learned from the owner's
  corrections.
- [Task forecasts and expected replies](TEMPORAL-FORECASTS.md): the retirement of task and execution forecasts;
  [declared reply forecasts](EXPECTED-REPLY-FORECASTS.md) still use the expectation ledger.
- [The durable autonomy scheduler](phase-c-durable-scheduler.md): leases, attempts and receipts for scheduled
  work; the scheduler and its store were later retired.
- [The situation and expectation spine](SITUATION-EXPECTATIONS-P6.md): the expectation engine; the situation part
  was removed.
- [Hands reply deadlines](hands-front-deadlines.md): how a turn that started a hand keeps answering.
- [Hardware-qualified latency gates](hardware-latency.md): local latency enforcement and hosted CI measurements.
- [Owner goal storage](owner-goals-storage-boundary.md): the boundary between goals and generic initiatives.
- [Suite verdicts](suite-verdict.md): offline runs, known failures and the optional daily timer.
- Related implementation records: [archive pull](../session-archive-pull.md),
  [device capabilities](../device-capabilities.md), [write origins](../hermes-write-origin-contract.md),
  [owner goal readers](../owner-goal-readers.md), and [secret reveal](../secret-reveal.md).

## Earlier agent studies

- [Agent screening benchmark](AGENT-BENCHMARK.md), the first development screen.
- [Native authority](AGENT-BENCHMARK-AUTHORITY.md) and [native memory](AGENT-BENCHMARK-NATIVE-MEMORY.md).

## Qualification packs

The packs `colonymind models packs` runs are described for users in
[portable behavioral packs](../QUALIFICATION-PACKS.md). These pages record how each pack was built:

- [Coding effects](qualification-coding-effects.md)
- [Source formation](qualification-formation.md)
- [Identity and audience](qualification-identity-audience.md)
- [Task transitions and interactive conversations](qualification-native-interactive.md)
- [Durable appraisals](qualification-perspective.md)
- [Function-router recovery](qualification-router-recovery.md)
- [Screening with an explicit output contract](qualification-screen-output-contracts.md)
- [Semantic recollection](qualification-semantic-recall.md)
- [Publishing serving cells](qualification-serving-publication.md)
- [Tool recovery](qualification-tool-recovery.md)
- [Shared work](qualification-unified-work.md)
