# Plasticity learners

## Service health (`service_health`)

The CPU service-health learner reads the instance's `usage.db` every five minutes. For each client and endpoint
alias with at least five calls it measures latency p95 in milliseconds, error share and call count. It also measures
the total file size under the instance directory, skipping links outside it. It keeps no request or response words,
endpoint URLs, host names or model names. It does not contact endpoints.

```yaml
plasticity:
  learners:
    service-health:
      implementation: service_health
      base: numeric-persistence@1
      kind: numeric-online
      teachers: {realized_value: value}
      model: {roles: [numeric_predictor], change: weights}
      budget: {type: anchored_output, bound: 0.5}
      effect: predict
      audience: owner
      personal_data: false
      erasure_path: rolling periods
      dependents: []
      mode: stack
      learning_rate: 0.01
      clip: 0.01
      gates: [retention, specificity, canaries, calibration, budget, finite, served_offline]
      arms: [persistence, seasonal_day, seasonal_week, ridge, time_shuffled, no_accumulation]
      comparator: {window_items: 200, window_hours: 168, bar: 0.02}
```

The five-minute interval, five-call floor, learning rate, clip, budget and comparator limits are plan assumptions.
The shared predictor has 12 lags, hour-of-day and hour-of-week sine/cosine features, and a constant. Each series
enters the predictor's `value` coordinate after subtracting its first observation and dividing by
`max(abs(first observation), 1)` (builder assumption). This mapping stays fixed in that alias's retained history;
the wire revision describes the predictor's standardised coordinate. Machine samples and histories expire after
14 days (plan assumption). A missing interval breaks pairing and warning streaks.

Each forecast goes through `Runtime.serve`; the next interval labels the same reference as `realized_value`.
Persistence is the first reference, followed by the previous serving closure. Daily
`comparator/numeric-YYYY-MM-DD.jsonl` files keep keyed references and paired finite numeric outputs and losses.
The numeric comparator withdraws a revision when its one-sided 95% lower bound on relative excess loss exceeds
the declared bar, after at least 30 labelled items (plan assumption). The newest item limit and age limit both
apply. Late outcomes cannot withdraw a newer revision. Files expire after 14 days, and comparator erasure rewrites
them without the erased references.
Erasure tombstones also prevent erased observations from returning after a disabled learner reloads. The current
runtime uses its generic eraser for unloaded learners, so physical deletion of numeric day-file rows in that case
finishes when the numeric comparator next loads; immediate deletion while unloaded needs the runtime integration
follow-up. Loaded numeric comparator erasure removes its rows immediately.

Nightly builds replay the bounded online rule in time order from the serving state. The newest fifth of fresh
timestamps forms an excluded temporal gate set, with at least 30 gate items (builder assumptions). Promotion needs
paired improvement against persistence and the previous closure, improvement over shuffled targets, finite weights,
the lifetime budget, refused invalid-input canaries, a refit calibration identity, and wire-reader agreement.
A failing night keeps the serving revision; measured live regression rolls it back automatically. The runtime's
existing compaction keeps 14 whole closures. Default parameters can fail to improve a particular stream; that is a
gate refusal, not a reason to promote it. Learning rates are declared per learner.

A surprise uses absolute forecast error divided by prior running mean absolute error, with a floor of `0.01`
(builder assumption). The first 30 scored samples only establish that scale. Two consecutive surprises of at least
4 post one `numeric_surprise` per subject, feature and UTC day through
`mind.observe_source("service-health", body)`. The generic summary expires in two hours. The observation path retains one upkeep
concern note, subject to its daily source cap. This path creates no task, message, hold, restart or configuration
change. With senses off, posts are ignored and counted; with plasticity off, no learner loads or samples.

The fixed-threshold arm warns after three consecutive intervals with error share at least `0.5` (plan assumptions).
Warning records retain that arm's timestamp and the learned warning's lead in seconds, including negative lead.
`colonymind report plasticity` reports streams, forecasts, surprises, notes, ignored posts, comparator results and
lead time. `colonymind plasticity comparator service-health` reads the same numeric comparison;
`colonymind plasticity rollback service-health` withdraws its current revision. Removing the declaration stops
sampling at the next service restart.

For other numeric learners, `plasticity.numeric.NumericLearner` implements `base`, `fit`, `losses`, `predict`,
`budget_used` and `exposure`. Its `TrainingRow.state` is a JSON numeric frame with `at`, `streams` (newest first)
and/or `inputs`; labels are scalar, output-ordered values, or mappings keyed by output name. Optional `previous_at`,
`gap`, `actuation`, `alarm` and `safety` fields fence learning. Declared safety values represent headroom and must be
at least twice their danger lines. Logistic learners use a constant anchor and a two-log-odds default lifetime
fence over standardised inputs in `[-3, 3]`, time features in `[-1, 1]` and constants at 1 (plan assumptions).
Factories call `numeric_comparator.configure(controller, declaration, variant="logistic")` before Runtime wires
their comparator; value variants provide output scales unless their outputs are already standardised. Numeric
`outcome(ref, observed, teacher="realized_value")` also accepts `verified_outcome` and `owner_label` as proof.

`numeric.save_revision(path, metadata, weights)` writes an immutable `numeric-revision-1` directory;
`numeric.load_revision(path)` checks the digest and supplies `Revision.forecast`. `W.npy` is C-order float32 with
shape `(outputs, features)`, and its digest is SHA-256 of the array bytes. `revision.json` uses `schema`, `learner`,
`variant`, ordered `features` and `outputs`, `standardisation`, `anchor`, `tz` and `weights_digest`. Stream lags,
declared inputs, time descriptors and the constant follow the shared contract. The service driver can export a
retained closure with `export_revision(closure, path)`; the core owns the retained weight copies and their erasure.





## Reply timing (`reply_timing`)

`reply_timing` is a kind D logistic learner for whether the owner begins a turn
within 60 minutes. It starts serving after its first successful nightly gates.
Its constant anchor and first live reference are 0.5. Every confirmed unprompted
send is paired with the previous closure on the same inputs; `NumericComparator`
automatically rolls back a measured Brier regression. The default comparison
uses the newest 200 labelled sends within 168 hours, a 0.02 relative excess-loss
bar and at least 30 outcomes (shared kind D assumptions).

```yaml
plasticity:
  learners:
    reply-timing:
      implementation: reply_timing
      kind: numeric-online
      base: reply-constant@1
      mode: rebuild
      audience: owner
      personal_data: true
      erasure_path: source erasure and clean rebuild
      effect: hold
      forced_run_share: 0.05
      dependents: []
      model: {roles: [numeric_predictor], change: weights}
      teachers: {realized_value: binary}
      budget: {type: anchored_output, bound: 2.0}
      learning_rate: 0.001
      clip: 0.5
      retention_margin: 1.0e-12
      gates: [retention, specificity, canaries, calibration, budget, finite, served_offline]
      arms: [constant, hour_mark, time_shuffled]
      comparator: {window_items: 200, window_hours: 168, bar: 0.02}
```

The input is hour of week and day, hours since the owner's last turn capped at
72, that turn's ledger channel, an outreach flag and a constant. The six most
frequent distinct last-turn channels in training have individual slots; all
others share one slot (plan assumption). The vocabulary is kept in the same
closure as the weights so rollback and erasure move both. The learner reads
only source ids, times and channel names, never conversation contents or
presence. A missing last turn supplies no forecast or training row.

Confirmed sends supply outreach rows. Hours on the UTC hour grid with no
outreach pending supply availability rows. Neither labels until its whole
60-minute window closes. Withheld or attribution-invalidated turns cannot
supply a label or a last-turn feature. Pending, failed, uncertain and requested
messages supply no reply-timing outcome. The row source declares both forward
`source_ids(ref)` and reverse `refs_resting_on(ids)` provenance, including the
outreach, label turns and last-turn feature. The shared runtime unserves and
rebuilds after erasure, including when this learner is disabled or absent.
There is no separate erasure listener.

Replay covers 90 days. Every fourth UTC week is permanently reserved for gates;
its rows never train. At least 30 reserved availability rows must show a clear
paired Brier gain over 0.5 and a serving-comparison lower bound at least
`-retention_margin`. This declaration defaults to `1e-12`, preserving the
original predicate, and must be a finite, non-negative number (booleans are
refused). In YAML, write a margin as `0.005` or `5.0e-3`; `5e-3` is read as a
string and refused. The anchor and shuffled-control lower bounds remain strictly above
zero regardless of this margin. The learner
must also beat a control averaged over eight target shuffles with seeds 7 to 14.
The control's reported baseline interval includes variation between shuffles.
Canaries, finite values, the two-log-odds fence and saved-state serving parity
must pass. The replay window, gate split, minimum sample count, shuffle count,
rate 0.001 and coordinate clip 0.5 are builder assumptions. Changing learning
settings requires a new learner identity. Availability supplies the promotion
check without relying on selectively sent outreach labels.

Every reply-timing gate invocation atomically writes a generic
`gate-certificate-1` record under the learner's `gate-certificates/` directory.
These survive runtime restarts and revision compaction. Records contain the
candidate and serving revision ids, gate-set digest, UTC time, Brier means for
candidate, serving, 0.5 anchor and averaged shuffled control, and paired
reference-minus-candidate mean, sample SD, SE and one-sided lower bound. The
anchor-versus-shuffled comparison also records shuffle SD, SE and its adjusted
bound. Counts distinguish candidate training rows from re-rendered control
training rows. All gate thresholds, check results, parity status
(`measured`, `not-tested`, or `error`), and gate outcome (`passed`, `failed`, or
`error`) are recorded before the controller considers promotion. A gate outcome
describes qualification, not whether the controller ultimately promoted it.
Insufficient rows leave statistics null and parity not tested; exceptions record
their type and propagate. Records contain aggregate evidence, never row inputs,
labels or references. Conservative source-erasure resets keep these statistics
and clear each record's `gate_set` digest.
Other learners can use `GateRecord` and `LearnerStore.certificates`; only reply
timing writes certificates currently.

Read certificates without loading a learner using this query. Supply the
configured learner name and a UTC interval `[START_UTC, END_UTC)` as ISO times
with an offset, for example `2026-10-14T00:00:00Z`:

```bash
python - reply-timing "$START_UTC" "$END_UTC" <<'PY'
import json
import sys
from colonymind.config import load_config
from colonymind.plasticity.certificates import GateRecords

config = load_config()
directory = config.home / config.get('plasticity.directory', 'plasticity')
records = GateRecords(directory / 'learners' / sys.argv[1] / 'gate-certificates').records(
    since=sys.argv[2], until=sys.argv[3])
for record in records:
    print(json.dumps(record, sort_keys=True))
bounds = [comparison['lower'] for record in records
          if (comparison := record['comparisons'].get('serving_candidate')) is not None
          and comparison['lower'] is not None]
print('Largest observed required margin:', max((max(0., -lb) for lb in bounds), default=None))
PY
```

Each serving comparison gives the required margin `max(0, -LB_serving)` for
that run. Choose any future margin from recorded evidence; the query only reads
certificates and never changes the declaration.

The learned hold applies after quiet hours, pauses, mutes, topic backoff,
budgets and the minimum gap. It may move a discretionary finding, loop or care
message toward an hour within two hours with at least 0.1 higher reply
probability, only while its expiry allows it. It stops two hours after the first
hold. Requested answers, follow-ups, duty, health notices, approval cards and
items due within two hours are excluded. First-held times live in `mind_state`
and are pruned after 24 hours. These time and probability limits are plan
assumptions. Nothing raises a budget or creates quiet hours.

A keyed five percent of would-be holds proceed immediately (plan assumption).
The recorded decision includes its sampling probability; observed outreach
rows use inverse-probability weights. Repeated ticks do not redraw that choice.
The ordinary outreach ranker and authority still decide whether a message goes.
No learner loaded, a base revision, or unreadable evidence means no learned
hold. A tick checks confirmed sends and closes outcomes; the live driver also
checks once per minute (builder assumption). Historical rows can train, but
sends before the learner first loaded are not presented as live comparisons.

`colonymind report plasticity` includes holds and forced runs over the last day,
served items and outcomes retained by the comparator, Brier score and log loss
against 0.5 and the per-hour mark, automatic rollback, and interruptions per
useful notice when the outreach run supplies usefulness verdicts. The mark arm
uses `0.5 * (1 - mark)` captured before sending (builder assumption). A reply
alone does not count as a usefulness verdict. `colonymind plasticity rollback
reply-timing` withdraws the current revision. Removing the declaration restores
fixed timing at the next restart. The separate dispatch reply forecast remains
at its existing fixed 0.5; contact timing, channel choice and presence are not
part of this learner.





## Action success (`action_success`)

`action_success` is a kind D logistic learner for a task's verified success.
Its constant anchor and first live reference are 0.7 (design baseline). After
the first successful nightly gates, its learned revision serves immediately.
Each dispatched task is compared with the previous closure on the same inputs;
the numeric comparator automatically rolls back a measured Brier regression.

```yaml
plasticity:
  learners:
    action-success:
      implementation: action_success
      kind: numeric-online
      base: action-constant@1
      mode: rebuild
      audience: owner
      personal_data: true
      erasure_path: source erasure and clean rebuild
      effect: predict
      dependents: []
      model: {roles: [numeric_predictor], change: weights}
      teachers: {verified_outcome: binary}
      budget: {type: anchored_output, bound: 2.0}
      learning_rate: 0.01
      clip: 0.5
      gates: [retention, specificity, canaries, calibration, budget, finite, served_offline]
      arms: [constant, task_type_prior, time_shuffled]
      comparator: {window_items: 200, window_hours: 168, bar: 0.02}
```

The learning rate, coordinate clip, two-log-odds budget and comparator limits
are implementation assumptions. Promotion needs at least 30 labelled training
rows and 30 labelled gate rows. Every fourth UTC week is permanently reserved
for gates; erasure cannot move its rows into training (sample counts and split
are builder assumptions). The base serves until promotion. Nightly gates check
paired Brier improvement over the constant, retention against the previous
closure, the time-shuffled control, canaries, calibration, budget, finite
values and saved-state serving agreement. A failed build leaves the serving
revision in place.

Features describe the task before its result: task type, drive, `open_ended`,
`goal_mode`, whether a `success_check` is present, standardised log runtime
minutes, retries allowed, recent failures on the same topic and their last
class (`failed`, `blocked`, `expired` or none), attached lesson count, hour of
week sine and cosine, and a constant. The 12 most frequent training task types
have individual slots represented by keyed hashes; the rest share `other`.
Prior failures cover 30 days and are capped at five; attached lessons are
capped at three (plan assumptions).
Task-type vocabulary and runtime scaling belong to the revision and move with
its weights on rollback or rebuild. `open_ended` comes from the candidate or
its recorded concern; legacy rows missing the flag use false (builder assumption).

The `verified_outcome` teacher labels a `mind:*` task as 1 only for `done`
verified by `owner` or `check`. It labels 0 only for `failed`, `blocked` or
`expired` verified by `owner`, `check` or `hermes_failure`. Other outcomes and
verifiers supply no label; a worker's unverified completion summary teaches
nothing. A verified blocked outcome without a timestamp becomes available when
first observed by polling or a nightly build, never retrospectively. Rows are
re-derived from source references for each rebuild. Stored
features contain categories, counts and times, with topics represented by
keyed hashes; titles, bodies, topic text and result text are never stored under
the learner. `source_ids` and `refs_resting_on` connect an intention and the turn
ids named by its context to its rows. Erasure unserves the learner and requires
a clean rebuild from the surviving sources before its next learned serve.

For discretionary task candidates the factor is
`min(1.0, max(0.5, p / 0.7))`, using that candidate's own served forecast.
This mapping is a plan assumption. Every other candidate receives 1.0,
including commitments, owed outreach and messages. The action-success factor
only lowers rank; the ranker's combined feedback and learned product remains
within `[0.25, 2.25]`. At the anchor the factor is exactly 1.0.

At dispatch, a forecast below 0.4 adds one line to the worker's task:
"A learned forecast puts tasks like this one at about N% success; check the
result against the task before reporting it done." N is rounded to the nearest
10 percent. The threshold and rounding are plan assumptions. The line appears
once per task. The learner adds no explicit hold, delay or ask, and never raises
priority or changes toolsets or authority. Lower rank can still put discretionary
work below the ranker's threshold.

Each dispatched task supplies one comparator item, referenced by its intention
id, with both the served probability and the reference closure's probability.
The first dispatch time, probability and `open_ended` flag stay with the primary
intention so repeated pulls cannot add a second comparison, even after the
comparator's 14-day retention expires (shared kind D assumption).
The live driver checks verified outcomes every five minutes (plan assumption)
and supplies an eligible label as `realized_value` to the comparator. The
newest 200 labelled items within 168 hours are compared; after at least
30 outcomes, a one-sided 95% lower bound on relative excess Brier loss above
0.02 withdraws the revision (shared kind D assumptions).

`colonymind report plasticity` shows forecasts served, notes given, factors
below 1, outcomes by verifier and class, Brier score and log loss against 0.7
and the task-type prior `(4 * 0.7 + successes) / (4 + n)` (design baseline),
using only earlier, unerased verified outcomes for that prior. Forecast, note
and outcome counts cover the comparator's retained 14 days; factors below 1
count retained intentions. Failed dispatches and breaker trips cover the last
seven days of the mind's counters (reporting windows are implementation
assumptions).
`colonymind plasticity rollback action-success` withdraws the current revision.
Removing the declaration restores factor 1.0 and no note at the next restart.
With no learner loaded, ranking and dispatch bodies use the existing path.





## Salience (`salience`)

`salience` is a kind D logistic learner for the owner's usefulness verdict on
acted discretionary intentions. Its factor is `clamp(p / p0, 0.5, 1.5)`, where
`p0` is the revision's frozen useful share of its owner-labelled rows.
The factor at the anchor is exactly 1.0. The ranker clamps its product with the
existing feedback multiplier to `[0.25, 2.25]`. Commitments, owed outreach,
check-in eligibility and drive weights keep their existing rules; a drive
weight of zero stays off. With no learner loaded, the existing formula applies.

```yaml
plasticity:
  learners:
    mind-3:
      implementation: salience
      kind: numeric-online
      base: salience-constant@1
      mode: rebuild
      audience: owner
      personal_data: true
      erasure_path: source erasure and clean rebuild
      effect: rank
      exploration_share: 0.0
      dependents: []
      model: {roles: [numeric_predictor], change: weights}
      teachers: {owner_label: binary, behavioural_proxy: binary}
      budget: {type: anchored_output, bound: 2.0}
      learning_rate: 0.001
      clip: 0.5
      gates: [retention, specificity, canaries, calibration, budget, finite, served_offline]
      arms: [constant, time_shuffled]
      comparator: {window_items: 200, window_hours: 168, bar: 0.02}
```

Exploration remains **0.0 while that decision is open**. The implementation accepts an
explicit share from 0 to 0.2; a keyed choice ranks that share of passes by
the existing formula and records the probability. Zero selects no such passes.
Each pass records at most 20 eligible discretionary candidates, their keyed
identities, factors and ranks under both closures, and whether they formed an
acted intention. It retains at most 200 passes for 14 days (builder assumptions).
This measures selection bias: outcomes exist only for acted items.

Features are the 12 most frequent keyed types plus other, drive, kind, concern
kind, open-ended status, source category, candidate salience and cost, drive
satiation, exposed worry and curiosity, hours since the owner's last turn
capped at 72, hour-of-week sine and cosine, and a constant. Rows keep categories,
numbers, times and keyed ids, never candidate or conversation words. The type
vocabulary, anchor and weights belong to the same revision. All admitted rows
count toward type frequencies; a changed vocabulary conservatively remaps
coefficients so a proxy or dismissal cannot raise a factor. The twelve slots,
72-hour cap, learning rate, clip and probability floor of `1e-6` for a
degenerate useful share are builder assumptions.

Only an owner's recorded verdict supplies an `owner_label`: `useful` and
`actioned` map to 1; `not_useful`, `wrong` and `dismissed` map to 0. An operator's
rating is not owner evidence. An acted intention without a reaction for 48 hours
supplies a `behavioural_proxy` push-away row. Ignored proxies do not count as
observed usefulness and do not change the anchor. Dismissals
count as zero in the anchor, but conservatively reducing the learned weights
and adding a nonpositive constant bias ensures that they cannot raise another
factor (builder assumption). The 48-hour window is a plan assumption. Owed and
unacted intentions provide no salience labels.

Source dependencies cover the intention and any source turns used by its
features or owner verdict. Withheld sources are excluded. The shared erasure
path withdraws affected revisions and rebuilds from retained source references;
weights and vocabulary rebuild together. Source erasure clears the propensity
log because ranks depend on the whole candidate set. If the driver is disabled
or absent, unresolved propensity provenance conservatively clears its derived
state and requires a fresh build when enabled again. Nightly promotion requires at least
30 reserved owner-labelled rows, paired Brier improvement against the frozen
anchor, no decline from the previous closure, improvement over time-shuffled
targets, canaries, finite weights, the budget and saved-state serving parity.

Each formed acted discretionary intention supplies one comparator item using
the served and reference probabilities captured for its ranking pass. The live
driver attaches the owner's later verdict with teacher `owner_label`; an ignore
does not become comparator proof. The numeric comparator rolls back a measured
Brier regression with the declared window and bar. The report includes factor
count and mean, shares below and above 1, outcomes by teacher, Brier and log loss
against `p0`, usefulness per acted concern, dismissals per week, the exploration
share and the selection-bias statement. `colonymind plasticity rollback mind-3`
withdraws the serving revision. Removing the declaration restores the existing
formula at the next restart.





## Contacts' reply timing (`contact_reply_timing`)

This is one pooled logistic predictor of whether a contact replies within the
declared 24-hour horizon after the agent writes to them. Its effect is `predict`:
it does not schedule a check-in, change backoff, hold outreach or send anything.
With no loaded learner, or at the base, a forecast is exactly 0.5 with the basis
`fixed uncalibrated 0.5 baseline`. A learned forecast names its serving closure
in `probability_basis`. Retries keep the probability that was originally served.

Its evidence is the comms ledger (`colonymind-comms.db`, `communications`). Each
row is read for its contact, direction, time, channel platform and source turn,
never its summary, and only conversation rows count: no delivery receipt and no
`mind:`, `system:` or `automated:` row (the same rule as
`CommsLog.history(conversation_only=True)`). The owner, placeholder ids such as
`default` and `system`, and deleted contacts are left out. A contact's rows are
one contact on every gateway: the sender is resolved to its canonical contact
when the row is written (one phone number is one contact), and a merge moves the
rows to the kept contact. A send is the last outbound row of a burst (outbound
rows to one contact at most five minutes apart, with no inbound between). It is
labelled 1 once an inbound row from that contact is recorded within 24 hours
after it, and 0 once 24 hours passed without one.

The inputs are computed from what was known at the send's time: the contact's
reply share `(replies + 1) / (sends + 2)` over the preceding 90 days of sends,
counting only replies recorded before the send, hours since their last inbound
(capped at 168), sends in the last seven days (capped at 10), six most frequent
training channels plus `other`, log horizon hours, UTC hour-of-week sine and
cosine, and a constant. These windows and caps are plan assumptions. Unknown
inbound history uses the 168-hour cap. A channel names its platform only
(`whatsapp`, `sms`, ...); any value that is not a known platform is `other`, so
a stored vocabulary never holds a handle. The pooled descriptors and weights
contain no contact identifier. Channel slots and their vocabulary travel with
each revision, including rollback and erasure.

Every 60 seconds the loaded learner serves each new send once, as it happens: a
send from before the learner started, or one whose reply is already recorded, is
history and is never served. A send served while its burst still grows (a later
outbound row joins the burst) is superseded by the burst's last row: only that
row is labelled, so a burst has one forecast and one label. The same pass labels
every served send once its label is known. A canonical reply forecast (`reply_forecasts`) still asks the
learner for its probability before it is issued and is labelled the same way,
from the comms ledger, within its own horizon. The row state contains numeric
features, timestamps, channel platforms and a keyed contact id, never message
contents or a contact's name or number. Both feature and label source
dependencies (the rows' source turns) are registered before learning or scoring;
a positive label also rests on the reply's source turn, so forgetting or
withholding the reply retracts it.
Source erasure unserves affected closures, removes comparator records and
rebuilds from clean references, including when the declaration is temporarily
absent.

A declaration can use the following settings. The learning rate, coordinate
clip, scaling, UTC split and gates are builder assumptions; the comparator's
200 items, 168 hours and 0.02 regression bar are plan assumptions.

```yaml
plasticity:
  learners:
    mind-2:
      kind: numeric-online
      implementation: contact_reply_timing
      base: contact-reply-constant@1
      teachers: {realized_value: binary}
      model: {roles: [numeric_predictor], change: weights}
      budget: {type: anchored_output, bound: 2.0}
      effect: predict
      audience: owner
      personal_data: true
      erasure_path: source erasure and clean rebuild
      dependents: []
      mode: rebuild
      learning_rate: 0.001
      clip: 0.5
      gates: [retention, specificity, canaries, calibration, budget, finite, served_offline]
      arms: [constant, time_shuffled]
      comparator: {window_items: 200, window_hours: 168, bar: 0.02}
```

Each nightly build starts at the constant anchor. Every fourth UTC week is
reserved for qualification; at least 30 training labels and 30 reserved labels
are required. The paired Brier gain must clear its one-sided 95% interval against
0.5, preserve the previous closure, and beat time-shuffled labels. Finite values,
the two-log-odds fence, missing-input canaries and serialized serving agreement
must pass. A passing revision serves immediately. One comparator item is recorded
per forecast with the served and previous probabilities, using 0.5 for the first
reference. Realized outcomes trigger automatic rollback on measured regression.
The report functions expose forecast and outcome counts, Brier score and log loss
against 0.5 over the comparator's retained 14 days (the shared plan assumption),
excluding erased records even when read without a running learner. Nonfinite
feature inputs are refused. A periodic callback retries missed outcomes without creating past
forecasts. Roll back with `colonymind plasticity rollback mind-2`, or remove the
declaration to restore the fixed baseline.





## Query adapter (`query_adapter`)

`plasticity.query_adapter.QueryAdapterLearner` implements the NumPy rebuild
protocol. It trains the contrastive query objective with in-batch negatives;
`arm="normalised_squared_error"` selects the comparison objective. Rank 32,
temperature 0.05 and the residual operator-norm bound 0.05 are design assumptions.
The module documents its training-rate, step-count and numerical assumptions.
Each step projects the factor product through its thin QR factors. Document
vectors stay fixed.

Owner labels and verified outcomes can train the global factors. Model teachers
train separate factors that adjust only a supplied shown set. Push-away targets
cannot become positive targets, and model teachers cannot raise adversarial
passages. `save_revision` and `load_revision` exchange immutable `query-adapter-1`
directories; shown revisions refuse global query application.

`plasticity.query_log.QueryLog` stores one owner query vector per turn, as float32,
only while a query adapter is loaded. Thirty-day retention is the plan assumption.
The row source must capture training dependencies before deleting queries, then
let source erasure unserve and rebuild the affected closures. Retained manifests
must still supply dependencies after the vector's retention period expires.

The registered `plasticity.learners.query_adapter` driver runs on the nightly
runtime. Host recall carries the authenticated owner decision and request turn id
through `LaneQuery`. Only that path applies learned factors. Contacts, guests,
explicit searches without a recall turn id, lessons, claim pairing and novelty
checks use the base query vector. The per-request transformed memo includes the
complete transform identity. With plasticity off or no declaration, recall keeps
its existing packet and cutoff behavior and creates no query log.

A declaration has this shape. Replace the base, dimensions and complete document
identity with the frozen index's recorded values; the example values are fixtures.
Rank 32 and bound 0.05 are design assumptions. The declared budget cannot exceed
that bound.

```yaml
plasticity:
  learners:
    recall-1:
      implementation: query_adapter
      kind: query-adapter
      base: embedding-base-revision
      dims: 4
      rank: 32
      document_identity:
        requested_model: example-embedder
        served_model: example-embedder
        declared_revision: example-revision
        dimensions: 4
        document_format: raw-text-v1
        normalization: provider-l2
        quantization: unspecified
        request_dimensions: null
      mode: rebuild
      audience: owner
      personal_data: true
      erasure_path: source erasure and clean rebuild
      effect: rank
      chain: recall
      dependents: [recall_cutoff]
      model:
        roles: [query_adapter]
        change: client_adapter
      teachers:
        owner_label: pairwise
        verified_outcome: pairwise
        stronger_model: pairwise
        deliberation: pairwise
        behavioural_proxy: pairwise
      budget: {type: operator_norm, bound: 0.05}
      gates: [retention, specificity, canaries, calibration, budget, finite, served_offline]
      arms: [identity, shuffled, normalised_squared_error]
```

Reference evidence keeps passage ids, canonical source ids and message hashes,
shown ranks, reranker scores and outcome dependencies. It keeps no text or query
vector. Training re-reads passage vectors from the existing index and query vectors
from the thirty-day log. The owner's retrieval lessons can supply corrections when
one cited correcting turn locates the immediately preceding recall in its session.
A corrected-to passage can be read from the index even when neither comparator
found it. Ambiguous lesson citations supply no outcome. The wider reranker teacher
uses at most twenty captured candidates and cannot label another chunk of a source.
No missing citation, unlabelled result, or unattributed useful-claim counter becomes
proof. Trusted resolver calls to `Driver.signal` and `Driver.outcome` require evidence
references; verified outcomes also require explicit attribution. The report lists
teachers that supplied no rows.

Nightly builds start from the identity on retained admitted references and fit a
seeded shuffled-pair control. Before running promotion, run
`colonymind plasticity qualify recall-1 --queries local-queries.json` to produce
`qualification/<state-digest>.json` under the learner directory. The command reads
the instance's existing index and uses its configured embedder and reranker;
`--embed-url` and `--rerank-url` override their endpoints while retaining their
configured models and credentials. It measures the candidate, base, previous
closure and fitted shuffled control through recall's collection and selection.
It does not promote a revision. The capture must
contain `state`, `control_state`, `query_transform`, `document_identity`, the
previous closure id in `reference`, `role: gate_suite`, a frozen `set_id`, current
provider `metadata`, and `development` and `held_out` score rows in the
`refit_query_adapter_cutoff` format. `gate` contains at least 1,000 distinct `ids`,
aligned boolean `reference`, `base`, `candidate` and `control` arrays, and
`canaries_base` and `canaries_candidate` exposure maps. Capture the actual results
of those revisions on the same inputs. Never relabel old scores for a new transform.
The scores carry no query vectors. The build and control are available through
`Driver.period()`, `Driver.build()` and `Driver.control()` for this offline job.
These sets are separate from sealed release evaluation sets, and cannot overlap
training turns, passage sources or label dependencies.

The query file is a JSON object with `role: "gate_suite"`, a stable `set_id`, and
four arrays: `development`, `held_out`, `gate` and `canaries`. Each row has a
distinct reference in `query_id`, transient `query` text, a `relevant` array of
canonical source ids, and boolean `no_answer` (true exactly when `relevant` is
empty). Each calibration split includes answers and no-answer queries; held-out
needs at least 150 no-answer queries, and gate needs at least 1,000 queries.
Canaries also name a nonempty `forbidden` array of source ids; their measured
exposure and correctness enter the capture. All query ids, including canaries,
are protected from training. Sealed sets, changed frozen sets, training overlap,
incompatible index identity and already captured or consumed revisions are
refused. `--max-queries` defaults to 10,000 selections across arms and
`--max-seconds` to 3,600 seconds (operator budget assumptions). Exhaustion writes
no qualification capture. Captures retain references, booleans and scores,
never query text, passage text or query vectors.

Both `benchmarks/public_memory/runner.py read` and
`benchmarks/source_recall/run.py` accept `--query-adapter REVISION_DIRECTORY`.
They apply a portable global revision to query vectors, check its document
identity, and include its digest in results. Document embeddings stay fixed.
Omitting the flag keeps the existing path; an identity revision preserves the
provider's query bytes. Shown-only revisions are refused by this flag.
`colonymind report plasticity` renders the faculty report's learner lines, and
`--json` emits the same collected report as JSON.

The driver consumes the held-out check durably before evaluating it, including
failed checks. It refits a cutoff on development scores and checks at least 150
disjoint no-answer queries once. Gate-suite retention, shuffled-control gain,
canaries, finite values, operator norm and serialized serving parity must pass.
A passing adapter, both factor scopes and its cutoff become one closure immediately.
A missing or failed qualification keeps the previous closure. The core discards
failed periods; do not run an unqualified build as a substitute for the offline
scoring job. The first serve also requires the release workflow's embedding-identity
check and sealed recall gates.

Each owner recall searches the vector lane under the serving and previous closures,
using the identity for the first comparison. `Runtime.serve` logs both top-k id
lists as keyed hashes once per turn. Membership of an owner-confirmed source in
each list gives paired boolean correctness. The existing live comparator rolls
back the joint closure on a measured regression, and the plasticity report includes
teacher counts, revision, cutoff and rollback events. No model teacher counts as
live proof. An unlabelled comparator reports that it is awaiting labels.

Erasure deletes query vectors and portable weight exports, resolves retained
reference dependencies even after query retention, then unserves and rebuilds through
the runtime. A recovery intent survives interruption between deletion and unserving;
completion clears it. An erased turn cannot be captured again. With no new qualified
clean closure, the existing core falls back to a clean retained closure or the base.
Operator rollback is `colonymind plasticity rollback recall-1`. Removing the
declaration restores the identity and the environment cutoff together.
