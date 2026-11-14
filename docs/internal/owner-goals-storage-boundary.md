# Owner-goals storage boundary

Owner records use `owner_goal_records`; all three approved kind names remain unchanged.
Owner audit uses `owner_goal_history`, including audit targeted at proposal IDs. Proposals,
artifacts and mutation receipts retain their dedicated `owner_goal_*` tables. Legacy answer
receipts and draft-acceptance joins move to `owner_goal_legacy_answers` and
`owner_goal_legacy_acceptances`. No owner record lives in `initiatives`.

`owner_goals.schema.initialize` runs before the initiative store is returned, even if the
owner API is disabled. A single SQLite transaction creates the tables, copies every legacy
column and stable ID, moves all related audit and joins, removes the shared copies, installs
retention and row-local rejection triggers, and writes the version marker. Proposals,
artifacts, receipts, tombstones and inherited tiers survive unchanged. A failure rolls the
whole transaction back; repeat opens perform a primary-key marker lookup. Generic writes
cannot insert or promote an owner kind, including direct SQL writers. Point mutations use
primary-key predicates and row-local kind checks, without membership scans.

The scan covered the entire sidecar for the table/view names, history and joins, database
filenames, dynamic table enumeration, snapshots, backup/export, metrics, and search. The
inventory below omits the retired initiative assignment helpers. The retained generic consumers are exercised
with a live Secret root, subgoal, progress and proposal. A source scan checks inventory completeness;
a call-profile assertion verifies every entry actually runs. The oracle checks returned
projections and all seven owner data tables after calls. Successful generic writes and
cleanup run alongside the hidden records.

## Direct SQL inventory

Paths are relative to `sidecar/colonymind`. Every listed entry is **unaffected (generic only)**:
its records and joins contain no owner rows after startup migration.

| Module | Reader or writer | Classification |
| --- | --- | --- |
| `init.py` | `_initiative_rows` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore._create_tables` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore._guarded_transition` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.answer_request` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.answer_request_write` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.answer_requests_since` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.count` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.count_transitions` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.create_intention` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.create_with_outcome` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.delete_old` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.failures_since` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.findings` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get_active_by_dedup_base` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get_agent_history` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get_by_ask_code` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get_by_dedup_key` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get_by_hermes_ref` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.get_history` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.intentions` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.last_responses` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.last_transition_at` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.lesson_rows` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.list` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.log_history` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.open_ask_codes` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.prune_intentions` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.retry` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.tokens_since` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.transition` | Unaffected (generic only) |
| `initiatives/store.py` | `InitiativeStore.update` | Unaffected (generic only) |
| `instinct/outcomes.py` | `Sources.history` | Unaffected (generic only) |
| `instinct/outcomes.py` | `Sources.intention` | Unaffected (generic only) |
| `instinct/outcomes.py` | `Sources.questions` | Unaffected (generic only) |
| `instinct/outcomes.py` | `_load_mind_wish_scope` | Unaffected (generic only) |
| `reports/instance.py` | `Instance.handed_over` | Unaffected (generic only) |
| `reports/instance.py` | `Instance.history` | Unaffected (generic only) |
| `reports/instance.py` | `Instance.mind_rows` | Unaffected (generic only) |
| `reports/instance.py` | `Instance.open_rows` | Unaffected (generic only) |
| `reports/senses.py` | `collect` | Unaffected (generic only) |
| `plasticity/learners/action_success.py` | `Rows.remember` | Unaffected (generic only) |
| `plasticity/learners/action_success.py` | `Rows.snapshot` | Unaffected (generic only) |
| `plasticity/learners/action_success.py` | `report` | Unaffected (generic only) |
| `plasticity/learners/reply_timing.py` | `Rows.snapshot` | Unaffected (generic only) |
| `plasticity/learners/reply_timing.py` | `report` | Unaffected (generic only) |
| `plasticity/learners/salience.py` | `Rows.label` | Unaffected (generic only) |
| `plasticity/learners/salience.py` | `Rows.snapshot` | Unaffected (generic only) |
| `plasticity/learners/salience.py` | `report` | Unaffected (generic only) |

## Dynamic walkers

| Module | Consumer | Classification |
| --- | --- | --- |
| `secrets/migrate.py` | `_tables` | Unaffected (generic only); excludes the owner-goals namespace |
| `secrets/migrate.py` | `StoreRewriter.walk` | Unaffected (generic only); uses `_tables` |
| `secrets/migrate.py` | `StoreRewriter._repoint_database` | Unaffected (generic only); uses `_tables` |

The oracle runs both dynamic walkers with a transform that would alter the Secret sentinel
if it reached an owner table. Maintenance cannot change owner revisions, receipts or audit.

## Store callers and projections

These consumers delegate initiative records/history to the inventoried boundary; they do
not need owner goals. Each entry is **unaffected (generic only)**.

| Consumer | Initiative usage |
| --- | --- |
| `api/routers/host.py` | Initiative CRUD, responses, context, collections and summary |
| `api/routers/mind.py` | Audit/search, dispatch, outbox, goals, asks, feedback and idempotent answers |
| `api/routers/secrets.py` | Mind note and permission-ask projection |
| `api/mcp.py`, `cli.py` | Generic mind/initiative route clients |
| `mind/audit.py` | Log, per-ID explanation, search projection and aggregate statistics |
| `mind/tick.py` | Formation, selection, dispatch, settlement, cleanup and health/metric counts |
| `mind/goals.py` | Agent goal budgets and steps |
| `mind/outbox.py` | Queued messages, delivery and full-text history |
| `mind/outcomes.py` | Generic results, ratings and answer settlement |
| `mind/authority.py` | Agent goal/task budget and recent-note checks |
| `mind/affect.py` | Agent action and outcome evidence |
| `mind/consolidate.py` | Agent finding/task consolidation |
| `mind/drift.py` | Generic action/rating comparison |
| `mind/lessons.py` | Generic lesson candidates and outcome labels |
| `mind/opinions.py` | Agent task outcome evidence |
| `mind/questions.py` | Question selection and settlement |
| `mind/wishes.py` | Wish-to-intention binding |
| `mind/observations.py` | Observation-generated generic messages |
| `mind/ask_origin.py`, `mind/permit_asks.py` | Generic ask history and origin |
| `mind/asks_report.py` | Initiative/history joins for the asks report |
| `hands/approvals.py`, `hands/controller.py` | Generic permit asks and approval history |
| `reports/faculties.py`, `reports/senses.py` | Report projections through `Instance` |
| `reports/bench.py` | Aggregate initiative qualification results; no owner-record hydration |
| `context_card/measure.py` | Opens the generic store for disposable measurement |
| `context_card/refresh.py` | Database revision signatures only; no record content |
| `memory/producers.py` | Read-only snapshot connections; callers retain generic table predicates |
| `server.py` | Initializes the store and its migration before consumers run |
| `qualification/native_unified_host.py`, `qualification/native_memory_worker.py` | Initializes generic stores in isolated qualification state |

Temporal-followup routes use the commitments database, not the initiatives table. Learners
other than the three direct-SQL modules above use their own learning stores. Source/vector
search and public-memory exports use the turn/source namespace; no initiative record or
history reader was found there.

## Owner consumers

These entries **need owner goals** and use the owner-goals store with an explicit audience.

| Consumer | Route and audience |
| --- | --- |
| `api/routers/owner_goals.py` | Calls `OwnerGoalsStore.read(owner, audience, ...)`; mutation context carries the server-bound owner audience |
| `owner_goals/store.py` | Owns all record/progress/subgoal, proposal, artifact and activity projections; checks `Audience` before projecting content |
| `owner_goals/artifacts.py` | Artifact resolution and run-output reads receive the explicit owner ID and `Audience` from the owner-goals store |

No generic learner, instinct evidence reader, report, mind tick, metric or search consumer
needs owner goals. Any future owner-goal integration must call this audience-bearing store
instead of querying owner tables or adding owner kinds to generic selectors.

## Recovery copies

`InitiativeStore.backup`, `init.backup_instance`, `backup._snapshot_databases` and
`backup.restore_full_backup` preserve whole SQLite databases for instance recovery. Twin
`SqliteSource._snapshot`, standby installation/merge and the operator SQL console likewise
operate under local instance administration. These are complete private state copies,
including credentials and owner tables, not generic evidence or audience projections.
The oracle snapshots a seeded store and runs the generic instinct/report readers against
that copy as well: they see no owner records or audit. Restored owner content continues to
use `OwnerGoalsStore` and an explicit owner audience. Recovery never flattens owner tables
back into `initiatives` or `assignment_history`.

## Regression checks

The HTTP existence oracle seeds an actual progress proposal for proposal detail and confirm
routes. Owner and absent IDs return the same result to unauthorized callers. The timing
oracle alternates Secret and absent IDs over 121 measured samples per target after warm-up,
with 10,000 generic rows, on guarded transitions. Median difference must be
below 0.2 ms; each path must also use fewer than 1,000 SQLite VM instructions. Migration
checks cover all three kinds, proposal audit, ordinary audit attached to owner records,
tombstones, engine metadata, artifacts, answer receipts, acceptance joins, rollback and
repeat opens, while retaining generic data.
