# Plasticity: learning in weights, governed like memory

Plasticity lets small models behind the agent learn from what happens: a decision model from the owner's labels and
verified outcomes, detached heads from what a sensor observes next, a query-side adapter from which memories were
used, a small predictor from a stream's next value. One rule protects every other guarantee: **weights learn habits
and salience; text keeps facts and authority.** A learned revision never decides who may do what, never holds a fact
the evidence ledger does not, and never serves an audience that could not see what it learned from.

This page is the contract of the plasticity core (`colonymind.plasticity`) and its instance wiring. The faculty ships
on with no learners configured, so a default instance does no weight learning. Configured learners run inside the
sidecar's existing nightly window. Every input is a protocol with a synthetic implementation, so the contract is
tested without a model service. The core holds no GPU code; NumPy is its only numeric dependency. A learner's own
computation (its steps, its budget measure, its predictions) belongs behind the `builds.Learner` protocol.

A night's learning uses the modules in this order:

| Step | Module | What it does |
|---|---|---|
| Who may teach | `teachers.py`, `objectives.py` | The closed teacher list and the objective registry |
| What arrives | `manifests.py` | Training rows by reference, the `RowSource` protocol, content-addressed manifests |
| What may train | `admission.py` | The admission rules, per row and per period |
| When | `periods.py` | The period ledger: open, closed, discarded |
| How it is built | `builds.py` | A clean rebuild from the base, or a stacked period |
| How far it may move | `budgets.py` | The declared budget per learner and the engine's recorded check |
| What serves | `revisions.py` | Revisions as closures, their atomic promotion, retention |
| Whether it may serve | `gates.py` | Evaluation hygiene, pooled suites, the statistics and the automatic gates |
| Where it stands | `states.py` | Candidate, serving, superseded, withdrawn; passing gates promotes immediately |
| Live comparison | `comparator.py` | The serving and previous closures scored on the same inputs, with regression rollback |
| Instance wiring | `nightly.py`, `rows.py` | Registered learners, the nightly cycle and Instinct's rows by reference |
| Compaction | `compaction.py` | Retained closures and pruning manifests of discarded periods |
| Whom it serves | `audience.py` | A stack serves only those who may see what it learned from |
| What it came from | `lineage.py` | Parent and child closures, their chain, the carry hook |
| Forgetting | `erasure.py` | Unserve, delete, rebuild or discard, with no contaminated file left |
| One learner's files | `store.py` | Manifests, periods, revisions, budgets and deltas together |
| Tests and drills | `synthetic.py` | A deterministic learner, its rows, its gate items and a driver for whole nights |

## Teachers and objectives

Every row names its teacher. `teachers.KIND_TARGETS` is a closed table of target types per kind:

| Kind | Allowed teachers and targets |
|---|---|
| `installed`, `detached-heads` | `next_observation`: vector |
| `query-adapter` | `owner_label`, `verified_outcome`, `synthetic`, `deliberation`, `stronger_model`, `behavioural_proxy`: pairwise |
| `numeric-online` | `realized_value`: value or binary; `owner_label`, `verified_outcome`, `behavioural_proxy`: binary |
| `supervised-refit` | `realized_value`, `behavioural_proxy`: label or binary; `owner_label`, `verified_outcome`: label, binary or distribution; `synthetic`: label; `deliberation`, `stronger_model`: label or distribution |
| `served-adapter` | `style`: label or distribution, in owner scope |

Model teachers never count as proof. Owner labels and verified outcomes count as live evidence; next observations
and realized values count on held-out frames or windows; synthetic rows count only in pooled gate suites. Style
and behavioural proxies cannot establish promotion evidence. Behavioural proxies only push away.

An objective declares a model, kind, target type, loss per teacher and blend. `objectives.register` checks these
against the closed table. Legacy underscore kind spellings remain readable so existing objective identities do not
change. Blend weights must be finite and nonnegative. Zero turns a teacher off; it changes neither the fitted state
nor another teacher's normalization. The blend is part of `Objective.identity`.

A learner implements `losses(state, rows, objective=...)`, returning one unblended mean per declared teacher, zero
when it has no rows. Both build modes store row counts and weighted terms. `objectives.teacher_lines(build)` returns
`teacher`, `rows`, `loss` and `counts_as_proof` (live evidence eligibility). The terms sum to `build.loss`.
The synthetic learner fits the sum of teacher means under the declared blend, so a larger teacher dataset does not
silently increase that teacher's weight.

`registration.check_learner(name, declaration)` returns a frozen `Registered` record, or raises `RegistrationError`
naming the failed check. Its required declaration fields are:

- `kind`; `teachers`, a mapping from teacher id to target type;
- `model`, with explicit `roles` and `change`;
- `budget`, with `type` and positive finite `bound`; nonempty `gates` and `arms` lists;
- `audience`, boolean `personal_data`, and a nonempty `erasure_path`;
- `dependents`, a list that may be explicitly empty; and `effect`.

Unknown values or missing fields needed for a check never pass silently. Protected roles cover identity, the owner gate,
authority, secrets detection, memory guards and floor patterns, speech voice, the live brain, live embedding and
reranking weights, and models used by qualified translators. Only client adapters may use the live embedding or
reranking path. The brain can use an offline served adapter; online changes remain refused, and a qualified
translator still protects its base. Roles come from trusted configuration, not model names or generated text. The engine separately
checks sites, kind-specific budgets, gates and arms, and runtime resource guards.

A decision-model declaration names its points. A head change names `own_head`; an encoder change lists every
`instance_points` member among its points. Safety points cannot share either learned change. The existing
objective admission rules still enforce tighten-only at consent and safety points.

Skip, hold and end-turn effects declare `forced_run_share`. `forced.ForcedRuns` uses an instance-keyed hash of the
learner and item reference to sample eligible items deterministically. Its default share is 5% (design assumption).
The sampling record includes the reference, eligibility, forced choice and observation probability. Only outcomes
of actual runs train; `outcome_weight` is their inverse observation probability. Noneligible items run normally at
probability one. The first skip learner will bind this record into Instinct's decision log.

## Training rows by reference

The decision log keeps no text. A `RowSource` re-renders each state from the row's references and yields a row only
when the state still matches the keyed hash logged with the decision, so an erased or changed source cannot
re-render and leaves the training set by construction. A row is a mapping with exactly these keys
(`manifests.ROW_KEYS`):

| Key | Value |
|---|---|
| `ref` | The decision id |
| `point`, `version` | The point and its version |
| `state` | The re-rendered state text, in memory only, never written |
| `state_hash` | The keyed hash logged with the decision (the de-duplication key) |
| `label` | The label from the strongest outcome |
| `teacher` | One of the teacher ids above |
| `push_away` | True when the teacher may only push away from the label |
| `source_kind` | Where the label came from (`explicit`, `resolver`, `behavioural`, `synthetic`) |
| `audience` | The narrowest audience the row's references show (`owner`, `contact`, `guest`) |
| `at` | The decision's time (ISO 8601 with its zone) |

A **manifest** is a period's rows by reference: every key but `state`, plus the admission's verdict for each row
(`admitted`, `witness`, `quarantine` or `refused`, with its reason), the count of witnesses, the rows each model
teacher had overruled and the count of malformed rows. It holds no text, and its id is a digest of what it lists.

## Admission

A row is admitted by its structure (its teacher, target, source, audience and time), never by its words. Per row,
in order, a row is refused when:

1. the objective names its points and the row's point is not among them;
2. its teacher is unknown, or the objective does not declare it;
3. its label does not fit the objective's target type, or, at a safety point, is not one of the point's labels (a
   soft distribution never trains a safety point);
4. its label came from an adversarial source kind (a contact's message, web or feed text, a tool's result): such
   streams are inputs, never labels, so they never label themselves;
5. a live row claims the synthetic teacher, or a synthetic row a live one;
6. it belongs to a protected set: a gate suite, a sealed evaluation set, a frozen holdout or a protected source.
   The protected refs are read at each admission, so a set registered later is refused too;
7. a push-only teacher claims to pull toward its label;
8. a model teacher pulls where the input is attacker-reachable (it may only push away there, or tighten a safety
   point);
9. at a safety point it loosens and its teacher is neither `synthetic` nor `owner_label`. Where the input is
   attacker-reachable (the point declares it, or the objective the row feeds does), only `synthetic` may loosen:
   the owner's false-flag marks tune calibration and seed synthetic sets, but never become loosening rows, because
   a crafted text that looks harmless to the owner would otherwise teach the screen to pass it.

A text-derived row younger than a day (D assumption) is not trainable yet (verdict `quarantine`), so corrections,
erasures and screening verdicts land first. Across a period's remaining rows:

- **Repeats are witnesses, not rows**: at most one row per state hash, per teacher, per period (the earliest). A
  state one teacher labelled two ways leaves the fit and is counted (`contradiction`).
- **The outcome wins**: where a verified outcome or the owner's label and a model teacher disagree on one state, the
  model's row leaves the fit and counts toward that teacher's measured error (`overruled`).
- **No single source dominates**: no contact, device or tool supplies more than 10% of a period's admitted rows (D
  assumption). A source's earliest rows are kept, so a later flood displaces nothing. The owner's own rows,
  synthetic sets and the system's resolvers name no source and are not capped.

A pair of ref and teacher that an earlier period decided is not offered again, so a witness stays a witness and a
dropped period's rows never train later. `Admission.readmit` admits a period again from what its source still
re-renders, at the original time: the manifest the period would have had if an erased row had never existed.

Recall targets carry structured evidence inside `label`, keeping the reference-row schema unchanged:
`preferred`, `disfavoured`, `shown` (passage ids to shown ranks), and `passage_sources` (ids to source kinds).
Missing provenance is refused. Recall deduplicates within a query and passage pair; outcomes override model
judgments only for that same pair. Deliberation and use signals train pairs only within the shown set. Their row
weight is multiplied by inverse preferred shown rank (a weighting assumption). A push-away pair only reduces
its disfavoured passage; it never rewards its preferred passage.

Only owner labels, verified outcomes, synthetic sets and a stronger reranker teacher's wider top 20 can reward
unseen passages. Model teachers only push down adversarial passages; synthetic rows cannot reward those passages
either. Only owner labels and verified outcomes may reward contact, web or feed passages. Wider reranker evidence
uses `model_role: reranker` and `reranked` (ids to ranks). Synthetic targets also name `question_id` and `haystack_sessions`; admission rejects
an overlap with any question or session in `evaluation_sets`. Callers supply the full evaluation registry. None
of this metadata is inferred from the passage's words.

## Builds

There are two ways to build a revision, and the ledger records both the same way.

- **Mode 1, rebuild** (`builds.rebuild`): where training data is kept by reference and retraining is cheap, each
  build is a clean retrain from the learner's base on the current manifests. An erased row cannot re-render, so it
  simply drops out of the next build.
- **Mode 2, stack** (`builds.stack`): where data is not kept or retraining is costly, each period trains from the
  state it starts from on its own rows only, and its delta is recorded. Erasure discards back to the last clean
  period.

Rows reach the learner in one canonical order (by ref, then teacher), whatever order the source yields them in, so a
build is a function of its manifests. A row the source no longer yields, or yields with anything admission decided
on changed (its state hash, label, direction, point, version, source kind, audience or time), is left out until its
period is admitted again: a row whose audience narrowed since would otherwise train at the wider audience it was
admitted with. A manifest admitted under another objective identity never trains: its period is admitted again under
the current objective first. A build's audience is the narrowest audience of what it learned, including
the state a stacked period started from.

## Periods and Budgets

**The period ledger** (`periods.py`) keeps, per learner, each period's life: it opens, receives its manifests and
closes; a stacked period also records the digest of its delta. A failed gate or an erasure discards it with the
reason, and a discarded period never trains again. Each period is one small file replaced atomically.

**Budgets** (`budgets.py`) bound how far a learner may move from its original. Each learner declares a budget type
(an absolute envelope, a fraction of each site's own norm, factor-norm bounds, an operator-norm bound from the
identity, or a bound on the output where the anchor is a constant or one-hot) and a positive finite limit. The
engine measures a learned state against it and hands the core a `BudgetCheck`; the core records it by the state's
digest. A check for a learner with no declared budget, of another type or against another limit is refused
outright; a check over the limit or not finite is recorded and fails the budget gate. A recorded check counts only
while the declaration it was made against stands: a tightened limit or another type needs a new check.

## Revisions

A revision is a **closure**: one id over everything tuned to a learned state, namely the base identity (weights,
quantization, tokenizer, engine version), the learned state's digest, the objective's identity (its blend
included), the calibration record ids and the declared dependents (a cutoff, baselines, a cache generation:
whatever the learner declares its revisions change). Outside its id it records what it came from: its parent, the
closures of other learners it learned from, its manifests, its periods, its audience and its build mode. Two builds
of identical bytes are one closure and keep the record written first: the id covers everything that serves, so that
record is a derivation of exactly what serves, and its audience and manifests answer for it.

- **Promotion swaps one pointer atomically.** A closure's state and record are each written whole first (an atomic
  rename after a sync); promotion then replaces the serving pointer in one rename. A crash before or during the
  swap leaves the old revision serving. A reader checks every closure against its id and every state against its
  digest, so no request ever serves half of one.
- **Each request reads one closure, for one requester.** A pin loads a closure and its state together, and names
  its requester: the base serves where the serving closure's audience does not allow it (the section
  [Audience](#audience)). A later promotion does not change what a pinned request reads.
- **Undeclared keys are refused.** A closure carries exactly its learner's declared dependents: carrying an
  undeclared key would change it silently, and dropping a declared one would serve without it.
- **The learner's own checks run before anything is written**: the manifests a closure names are kept and belong to
  the periods it names, and it claims no wider audience than what it learned from.
- **A new base is a new closure family.** A store opened on another base identity serves its own base, never a
  closure learned on the old one; the old family's states become orphans, which the next sweep removes.
- **The newest 14 closures are kept whole** (D assumption), the serving one always among them. An older one is
  dropped with its state file and leaves a tombstone: its record without its state, naming only ids, so lineage
  stays walkable and an erasure still reaches what was built from it. Tombstones never serve.

## Gates

**Sets are kept apart.** Nightly selection runs on gate suites; release claims use sealed evaluation sets that no
selection ever touches, since repeated selection against one set would leak into it. The set ledger
(`gates.SetLedger`) enforces it by set id and a use ledger: a set used for selection can never be named for
evaluation, a sealed set is never used for selection, a gate suite and a sealed set share no item, a set's role
and items never change, and every registered item joins the protected set that admission refuses. A suite gates
only when every one of its items is registered under its id.

**Pooled suites.** A gate suite pools at least 1,000 items from templates at new seeds (`gates.pooled_suite`); a
frozen seed or a repeated item is refused, and a smaller suite cannot gate. The gates count distinct items only
(`gates.check_distinct`, by id and by state, however the suite was made): a repeated item is one observation counted
many times, and would tighten the bound on nothing new.

**The statistics.** Non-inferiority is judged by the one-sided 95% upper bound of the decline, never by a point
estimate. Over paired items the decline is (losses minus gains) over n, where a loss is an item the reference got
right and the candidate wrong. Its bound is the decline plus 1.645 standard errors, with the variance floored so
that a set with no discordant pair still carries the rule of three (3 over n). This reproduces the design's
arithmetic: 100 items with no error at all still leave a 3-point bound, and 1,000 items at 3% disagreement with no
net decline give about 0.9 points. The floor is my construction (my assumption); it never tightens as losses grow.

**The gates** (`gates.evaluate`), each a named check:

| Gate | Passes when |
|---|---|
| `retention` | The decline's bound is at most 1 point (D assumption) against the serving revision and against the base, so nights that each decline a little never add up |
| `specificity` | The candidate beats its shuffled-label control: the lower bound of its gain is above zero (my assumption for "beats") |
| `canaries` | At least one canary is planted, and every one's exposure is at the base's level, never above it |
| `calibration` | The calibration was refit for this revision through the caller's refit hook; thresholds never carry over from other weights |
| `budget` | The engine's recorded budget check of this very state passed |
| `finite` | Every learned value and every metric is finite |
| `served_offline` | The served outputs agree with the offline ones, to 1e-6 (my assumption) |

A failed gate drops the period, and the previous revision keeps serving. Nothing waits on a person.

## States and promotion

A revision is a `candidate` while its build is checked, `serving` when the pointer names it, `superseded` when a newer
revision has replaced it, or `withdrawn` when a failed gate, rollback or erasure has pulled it. The check is part of
the nightly cycle, with no separate observation period or first-promotion approval state. `states.Controller` moves
the revision through these states:

- **Automatic promotion inside the gates.** A candidate whose gates all pass is promoted at once, including the
  learner's first revision. A failed gate
  withdraws it and discards the periods it introduced (`RevisionStore.introduced`: those no earlier closure that
  was never withdrawn names), whether or not the caller names the period.
- **Automatic rollback.** A later gate that fails on the serving revision withdraws it, discards the periods it
  introduced, and serves its nearest retained ancestor that served before, else the base. A discarded period never
  trains again, so the next build does not learn the failed data back, and a candidate built on such a period
  before it was discarded is withdrawn when it is considered, whatever its report.
- **A comparator follows live traffic.** Every served item is evaluated by both the serving closure and the previous
  closure, or the base for a first revision, on the same inputs. The log names the learner, both closures, the item
  reference and both outputs as values, declared labels or keyed hashes. Text and raw inputs never enter the log.
  Outcomes attach later by reference. The rolling window keeps up to 200 labelled items or 7 days, whichever comes
  first (my assumptions). A measured regression rolls back when the one-sided 95% lower bound of the decline exceeds
  the learner's declared bar. The report includes both bounds: the upper bound used by offline non-inferiority
  gates is not by itself proof of a live regression. A withdrawn closure never serves again, and its discarded
  periods do not train the next revision.

Promotion, refusal and rollback each leave a report event with the closure ids and reason. Erasure and rollback use
the same learner or chain lock; a rollback cannot race the clean rebuild.

## Audience

A row's audience is who may see what it was learned from: `owner` (the owner alone), `contact` (the owner and the
contacts it concerns) or `guest` (anyone). **A stack's audience is the narrowest audience of its training data**, so
a stack that learned from one owner row is an owner stack. A requester is served a stack only when it may see
everything the stack learned from:

- an owner stack serves owner turns only;
- a contact stack serves owner turns only too: the core cannot tell which contact a stack's rows concern, so no
  contact is served what another contact's rows taught;
- a guest stack serves anyone.

Everyone else gets the base (for a query adapter, the base is the identity map). An unknown requester, or a stack
whose audience is not one of the three, gets the base. No request path reads the serving stack without naming its
requester: the revision store's own pin applies this rule, and `audience.AudiencePolicy` applies it with the
learner's named exception.

A **named exception** (`audience.AudienceException`) is an explicit per-learner declaration that its stack may also
serve the audiences it names, with the reason it is safe (a model that outputs only a label from a closed set, for
example). It is bound to the objective it was reasoned about: a stack trained toward another objective is not
covered until the learner declares it again. A learner that may train a safety point never takes one, and neither
does a learner whose objective names no points (it may train any).

## Lineage

A closure's parent is the closure it was built from; its upstream closures are other learners' closures it learned
from (a score head trained on a query adapter's candidates). `lineage.Lineage` walks both, tombstones included, so a
pruned closure in the middle of a chain never hides what was built from it: erasing a row that trained a parent
reaches every descendant, in the learner and across learners.

**The carry hook** binds a learned child's promotion to Instinct's per-point decision state, with the child's own
calibration refit. The nightly controller calls it with the parent and child. The carry uses a compare-and-set of
the tier state: a demotion that lands while the gates are read, or between the final read and write, wins. Carries
are shown in the nightly summary and Instinct's command line.

## Erasure

The guarantee: **the erased item is removed from every retained training artifact, and no served revision was
trained with it.** It does not undo actions the agent already took. Given erased record ids, `erasure.Erasure`:

1. **finds** every manifest that names an erased ref, the periods that admitted one, and every closure trained on
   one (whole or as a tombstone), for a stacked learner every closure of a period that will be discarded, and every
   descendant of any of them;
2. **unserves at once**: a learner whose serving closure is contaminated falls back to its newest retained closure
   that is clean, served before and was never withdrawn, else the base. A chain of learners qualified together
   falls back as one, to its newest joint record whose members are all clean, else every member to its base, never
   to a mix. Its members change one pointer at a time, so the move is recorded before the first changes and cleared
   after the last (`JointLedger.begin` and `end`), and a request reads the chain as one (`JointLedger.pin`: a
   recorded joint closure whose members' audiences all allow the requester, else every member's base). A crash
   between two member swaps therefore never serves a mix, and the next erasure (an erasure of no refs will do)
   finishes the move; it also moves back to its newest clean joint record a chain whose members serve a tuple no
   joint record names;
3. **deletes** the contaminated closures, their states, statuses and budget checks, and the joint records naming
   them. Each closure goes after every closure built from it (`Lineage.leaves_first`), and the manifests and
   periods that name a ref change only after that, so whatever a crash leaves is still reached on a retry. A rebuilt
   learner's manifests are rewritten without the refs (admitted again from what the source still
   re-renders, so a witness of an erased row becomes the row); a stacked learner's periods are discarded from the
   first one that trained on a ref, with their deltas;
4. **rebuilds** each rebuilt learner through the caller's rebuild (a clean retrain on the rewritten manifests, gated
   like any candidate). A stacked learner's data was not kept, so its discarded periods are gone. A retrain that
   derives a closure withdrawn before (an erasure of every row a learner had learned from can return it to a state
   an earlier night withdrew) proposes and promotes nothing: the learner keeps serving what step 2 left it, and the
   erasure goes on to step 5. The runtime's deferred rebuild of a learner that was not loaded during the erasure
   (`rebuild_required`, run when it loads again) completes the same way, removes its marker and leaves the learner
   serving the base;
5. **sweeps and scans**: the temporaries and orphans a crash left are removed, so a retried erasure finishes what an
   interrupted one began. Every store is then scanned for the erased refs, for state hashes no surviving row shares
   and for any orphan still on disk; anything found is reported. What the erasure did is recorded by count and
   closure id, never by the erased refs.

Instinct's forget path supplies the forgotten decision refs before deleting its rows. Calibration records naming
removed learned revisions are withdrawn, so the affected point abstains until a fresh refit. An interrupted erasure
finishes on a retry with the same refs. The erasure drill checks the rebuilt state against a clean retrain without
the row, within 1e-5 relative (D assumption). An erasure, rollback and build of the same learner never run at once.

## The store

`store.LearnerStore` keeps one learner's files under `learners/<name>/`: `manifests/` (content-addressed),
`periods/`, `revisions/` (closures, tombstones, learned states, statuses and the serving pointer), `budgets/`,
`deltas/` (stacked periods' deltas) and `erasures/`. Every file is written through one atomic write: the bytes go to
a temporary beside the target, are synced, and replace it in one rename, after which the directory is synced. A
crash leaves at most a stray temporary, which the next open or sweep removes and no reader opens.

## The synthetic learner

`synthetic.py` holds what the tests and drills run on: `SyntheticLearner`, a deterministic logistic regression over
synthetic features in NumPy that implements the learner protocol (a push-away row's term is a margin on the
disfavoured label only); `SyntheticSource`, a row source over rows in memory that stops yielding an erased row and
one whose state no longer matches its hash; `SyntheticTemplate`, gate items at a seed; planted canaries whose secret
label only memorising could expose; and `Cycle`, which drives one learner through whole nights (admit, build,
propose, gate, consider) the way a caller would. One end-to-end test drives it through observe, admit, build, gate,
promote, erase and rebuild.

## Numbers

Every number here is an assumption, not a rule. *D assumption* marks a number the plasticity design chose; *my
assumption* marks one this build chose.

| Number | Where | Label |
|---|---|---|
| Text-derived rows wait 24 hours before they train | `admission.QUARANTINE` | D assumption |
| One source supplies at most 10% of a period's rows | `admission.SOURCE_SHARE` | D assumption |
| The newest 14 closures are kept whole | `revisions.KEEP` | D assumption |
| A decline of at most 1 point, by its one-sided 95% bound | `gates.DECLINE_BAR` | D assumption |
| A live window of at most 200 labelled items or 7 days | `comparator.py` | My assumptions |
| Pooled suites hold at least 1,000 items | `gates.MIN_POOLED` | The design's arithmetic for a 1-point bound |
| The variance floor that carries the rule of three | `gates.decline_bound` | My assumption |
| "Beats its shuffled control" means a gain whose lower bound is above zero | `gates.evaluate` | My assumption |
| Served and offline outputs agree to 1e-6 | `gates.SERVED_TOLERANCE` | My assumption |
| A canary may not exceed the base's exposure at all | `gates.CANARY_TOLERANCE` | My assumption |
| The core objectives' blend weights (1.0 for outcomes and the owner's labels, 0.5 for synthetic rows and deep answers, 0.3 for model teachers on recall, 0.1 for proxies) | `objectives.py` | D assumption |
| A rebuild matches a clean retrain within 1e-5 relative | the erasure drill's tests | D assumption |
| The synthetic learner's features, steps, rate, budget, suite and canaries | `synthetic.py` | My assumption, for tests |

## Wiring

`mind.faculties.plasticity` ships on. With no configured learners the runtime loads nothing, schedules no learner
work and reports that it is idle. Switching the faculty off makes every request read the base and prevents nightly
learning. The top-level `plasticity` section holds the state directory, under the instance's data directory, and a
`learners` mapping. Each declaration passes `registration.check_learner`; one failed check prevents that learner
from loading, names the failure in the report and leaves the other declarations independent.

The nightly runtime observes rows, admits them, closes each period, builds by the learner's declared mode, gates,
promotes or drops, and compacts. Re-rendered decision states stay in memory; manifests hold references and keyed
hashes. A qualified chain reads through `JointLedger.pin` and promotes as one joint record. Compaction keeps the
newest 14 whole closures (D assumption), including the serving one, and prunes manifests no retained period needs.

A closure id covers only what serves (base, learned state, objective, calibration, dependents), so a night that adds
nothing new to what a withdrawn closure learned derives that closure again. The night reports it as `unchanged` with
`why: ["withdrawn_before"]`: nothing is proposed, gated, promoted or rolled back, and serving stays as it was. Its
period is kept only where a later build can still learn from it: when it holds quarantined rows (offered again once
trainable, while every row of a discarded period is excluded for good), or admitted rows of a learner whose every
build reads every retained period (`mode: rebuild` with accumulation on). Any other period is discarded: one with no
admitted or quarantined row, and one of a stacked or non-accumulating learner (whose build reads only the newest
period) with no quarantined row.
`Controller.propose` still refuses a withdrawn identity (`WithdrawnClosure`) for every other caller.

A chain moves only as one, and a night it cannot move judges no member for the chain alone. When a member re-derives
a withdrawn closure, every other member reports `unchanged` with `why: ["chain_waiting"]`, whatever its own gates
said. When no member does but one fails its own gates, that member is judged as usual (dropped, its new periods
discarded) and each member that passed its own gates reports `chain_waiting`. A waiting member's candidate stays a
candidate, never withdrawn, so the same derivation can promote on the night the chain moves; its period is kept when
the candidate names it, and otherwise by the rule above. A member whose serving closure fails its gates still rolls
the chain back. When a learner's next closure is judged, the candidates it left waiting that this closure replaces
are deleted, not withdrawn: they never served, so they do not count as having vetted the periods they name.

A night that fails reports the exception type and, for a refused state change, its fixed code (`audience_wider`,
`report_mismatch` and so on), never the message.

**Kind D in the sidecar.** The NumPy `numeric-online` learner supports persistence-anchored values and
constant-anchored logistic probabilities through the shared `numeric-revision-1` format, with bounded online steps,
lifetime output fences, temporal pair breaks, control arms and numeric metrics. Its registered numeric comparator
records paired finite outputs and realised losses in daily files and automatically withdraws measured regressions.
The `service_health` implementation samples local usage and storage, serves forecasts immediately, qualifies learned
revisions nightly, and sends warnings only through the senses observation edge. Its declaration, report fields and
assumptions are in [Plasticity learners](PLASTICITY-LEARNERS.md#service-health-service_health).

The instance command line exposes the same public state as the report:

```text
colonymind plasticity status --json
colonymind plasticity learners --json
colonymind plasticity revisions <learner> --json
colonymind plasticity comparator <learner> --json
colonymind plasticity rollback <learner> --json
colonymind plasticity erase --refs <record-ref> ... --json
```

Every command also prints a plain-text result when `--json` is omitted. Promotion is automatic, so there is no
promotion command. Each learner's report line separates rows and loss by teacher, states whether each teacher
counts as proof, counts promotions, refusals, rollbacks and erasures, and explains the live comparison. Missing
teacher reporting is named explicitly instead of inferring proof from a model's labels.

Paired arms isolate one change at a time: `full-plasticity` puts all learners at their bases;
`plasticity.<learner>` puts one at its base; `plasticity.instinct`, `.recall`, `.perception`, `.telemetry`, `.timing`
and `.voice` put an area at its bases; `plasticity.<learner>.teacher.<id>` drops only that teacher.
`shuffled-control` uses shuffled training targets, and `no_accumulation` removes learning across periods. The
faculty arm `all-on-plasticity` switches the faculty off. Learner and teacher names come from the declarations.

The faculty report dispatcher includes the plasticity reader in `colonymind report faculties`. Paired plans accept
a shared `plasticity` configuration through `--plasticity-config`, freeze its learner declarations in every arm,
and expose the declared learner and teacher interventions through profile validation. Disposable benchmark workers
apply those transformations to the instance configuration and run configured learners at the nightly boundary
as the episode's body clock advances. Erasure removes calibration records from the store and publishes a shared
cache generation; Instinct engines in the same or another process check it before reusing cached records, so the
next read drops erased calibrations immediately.

## Behaviour and standing lessons

The `behaviour` kind teaches through existing lessons: `effect: teach`, model
role `standing_lessons`, change `context`, and a three-level `ladder` budget.
It admits no training rows and updates no model weights. Core enables
`behaviour-tool-failure` with a reflector source, idle pending reflector lesson admission and selection.
Core contains no lesson text or style patterns. Hosts can override fields or
set `enabled: false`.

An optional host declaration for a long-dash target follows. The owner provides
the instruction in their own channel; normal verified lesson admission records
it on the declared topic. Until then the driver returns `unchanged`, reason
`awaiting_owner_lesson`. No relayed copy is created.

```yaml
plasticity:
  behaviour:
    restarts: []  # UTC change timestamps for head, stack or host-agent changes
  learners:
    reply-punctuation:
      implementation: behaviour
      kind: behaviour
      effect: teach
      base: standing-none@1
      mode: rebuild
      teachers: {verified_outcome: binary}
      model: {roles: [standing_lessons], change: context}
      budget: {type: ladder, bound: 3}
      audience: owner
      personal_data: true
      dependents: []
      erasure_path: "reads live owner turns; examples erase with their source; certificates contain counts"
      gates: [lesson_current, standing_cap, budget, finite, served_offline, baseline_target, baseline_guards]
      target: {signal: long_dash, rank: 1}
      lesson: {source: owner, topic: long dashes}
      contrast:
        - {pattern: '\s*[\u2014\u2013\u2015\u2e3a\u2e3b\ufe58]\s*', replacement: ', '}
      baseline:
        days: 10
        min_replies: 60
        min_session_days: 12
        target_cap: 0.03
        guards: {substitute_rate: 0.02, median_length: 0.15, tool_failure_rate: 0.01, owner_corrections: 0.10}
```

Baseline counts freeze once per lesson at its first promotion, from the preceding
days. The learner state stores the lesson id, timestamp, aggregate counts and pass
history; ladder promotions and process reloads preserve them.
Taught replies accumulate from promotion onward. Both populations use
session-day clustering and Wilson/Newcombe one-sided 95% bounds. A pass needs
the taught share at or below the cap and a positive lower bound on baseline
minus taught. Below either minimum, `awaiting_evidence` leaves the closure
serving. A measured reversal rolls back; a target miss after a pass also rolls
back. Once minimums are met, a taught rate above the cap fails and withdraws
the level even when it improves on baseline. Other inconclusive targets before
a first pass keep accumulating evidence.
Measured guard breaches roll back and halt the ladder. Inconclusive guards do
not block. Reports and certificates include platform breakdowns.

Additional configurable `baseline` defaults are `standing_days: 14`,
`bootstrap_samples: 1000` and `min_guard_units` (assumption: defaults to the
declaration's `min_session_days`). The bound is fixed at one-sided 95%.
Guards below the session-day minimum in either
population are `not_measured`. Median length uses a deterministic cluster
bootstrap. Windows and minimums are positive integers; margins are finite and
nonnegative. Every `not_measured` guard is named in certificates and nightly
results. The tool guard divides teachable failures by attempts (calls plus parse
failures), including every turn with operations. Appending a timezone-aware
ISO-8601 host change timestamp resets the taught window while preserving the
baseline and pass history. Certificates record each restart and label the residual
confound: the frozen baseline predates that head/stack change. This comparison
cannot isolate changes the host has not declared. No randomized assignment or
forced-run share is used; baseline and taught name the two measured populations.

Every level begins with a structural trial certificate. Level 1 carries the
instruction. Level 2 adds a live contrast when the host declares replacement
patterns. Level 3 moves the section last. A failed level withdraws to base
before another level is proposed; level 2 is skipped without patterns.
Exhaustion returns `ladder_exhausted`. Examples retain a turn reference and
sentence offsets, never copied prose; erasure removes the reference and
rendering returns to the rule alone. Certificates contain aggregate counts.

At most two lessons appear in the pinned `Standing lessons` section on every
eligible owner turn. The adapter always sends that section in full. Existing
JSON use context records `turn_id` and `standing: {learner, level, closure}`.
Task bodies also carry standing lessons. They bypass meaning matching,
cooling, review retirement and night supersession; owner retirement and source
erasure take effect next turn. The meaning path retains its one-lesson limit
and excludes standing lessons. Relay sources use the existing provenance-aware
relay route; reflector sources remain idle until lesson admission and selection are implemented.
