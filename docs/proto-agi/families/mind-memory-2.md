# mind-memory-2: pre-registered plan for the harder memory family

Historical plan: `context.blocks`, `minus_blocks`, `full-blocks` and `all-on-blocks`
have been retired. Current runs always fit context and enforce its item, character
and time caps. The block-switch contrasts below describe the earlier protocol.

Status: pre-registered plan for a second memory family, written from the dev templates alone,
before any pilot. `mind-memory-1` states every fact a few turns before its probe, where stock
Hermes's own session memory reaches it as easily as ColonyMind's recall, so that family leaves little
room for a gain, and a saturated family cannot show one. This family moves the facts into a long seeded history, states each one once and many sessions back,
and surrounds it with near-duplicates, including the assistant's own restatements. The pilot fields
are placeholders until the paired dev pilot runs; the gate's own numbers land in the run's report,
never here.

## 1. Hypotheses

- **Memory.** Hermes with the ColonyMind mind (`full`) answers questions about facts the owner
  stated once, many sessions back, in another channel, later corrected, or restated wrongly by the
  assistant, and abstains when the history holds only near misses or a withdrawn fact, more often
  than stock Hermes (`base_hermes`), on held-out scenarios of this family.
- **Memory blocks.** The resident-block fit (`context.blocks`, on by default) costs no recall on
  long histories: `full` is non-inferior to `full-blocks` (the per-turn selection alone).
- **Learning (guard only).** Turning the learning rules on (`full-plus-learning`) costs no recall
  here. This family cannot show a learning gain (section 8, item 1); it guards against a loss.

## 2. Family

| Item | Value |
|---|---|
| Dataset id / version | `mind-memory-2` (generator protocol `paired-generator-1`, episode grammar `paired-workflow-runtime-1`, history `paired-history-1`) |
| Dev templates | `benchmarks/paired/generators/memory_hard.py` (`--family memory_hard`): 7 `recall` (`buried-fact`, `buried-correction`, `channel-switch`, `assistant-misquote`, `restated-stale`, `buried-preference`, `spread-total`), 2 `abstain` (`near-miss`, `retracted`) |
| Dev split for the pilot | seed 7, `--per-template 3`, 27 episodes (21 recall, 6 abstain); loader content hash `a463ef8d2bd874f84cd7a470f9db48b99ecb63c7b6eebca7ccf3399157b3f804` (`scenarios.json` sha256 `167e2318b9748a9a401fb52bd39621ac9883dfebba43526db64c6a9e24a280bd`), rendered once into a private directory |
| Held-out templates | A Python module **outside the repository**, written by someone other than the author of the dev templates from a schema-only brief, named at plan time by `COLONYMIND_HELDOUT_TEMPLATES` (or `--heldout-templates`); the generator refuses a path inside the repository and the file is never committed. It declares `FAMILY = 'mind-memory-2'`, covers the nine types with phrasings, vocabularies and history filler not written to match the dev templates, keeps the two abstention types as hard controls, and keeps the shape below (history length, depth of the buried statements, one probe after a night and a restart) |
| Held-out rendering at the gate | a fresh 32-bit seed chosen at plan time, `--per-template` set so the split holds at least `n` scenarios, into a fresh private directory whose content hash the plan freezes; never re-rendered over; a failed gate is re-tested only on fresh instances |
| History | 28 to 40 earlier sessions (`history`), dated from a fixed day and one to three days apart, each holding its planted exchanges among 2 to 4 filler exchanges (an owner turn and the assistant's reply); session ids name the channel (`chat-NN`, `email-NN`, `sms-NN`). The filler states unrelated facts of the same kinds (where things are kept, who borrowed what, where a monthly review is held, boxes of files dropped off) and talk that states none; it never contradicts itself or the template's facts |
| Episode shape | one neutral owner statement (`owner-1`), one night crossed (`advance_clock: 86400`, then `tick: 1`), a process restart (`workflow.restart_before: [3]`) and the probe in a new session (`owner-2`), the only request in the episode, asking for `answer.json` |

What each template makes hard (the family's tests assert each property on the rendered history):

| Template | Group | The fact | The trap | Oracle |
|---|---|---|---|---|
| `buried-fact` | recall | a place, stated once in the first four sessions | the item comes up again without its place; more than twenty sessions follow | `place` |
| `buried-correction` | recall | a monthly review's venue, changed once at least fifteen sessions later | the same review with another contact and the contact's review of another item, in other venues | `place`; the old venue forbidden |
| `channel-switch` | recall | who has an object now and where, in a terse lower-case text-message session | an earlier loan to another contact in an e-mail-style session, then its return; the probe asks in prose on another device, in other words | `contact`, `place`; the earlier contact and place forbidden |
| `assistant-misquote` | recall | where the owner said a thing goes | the assistant restated it with a near-duplicate wrong place in its reply and twice more in later recaps; the owner never repeated it. Recall that ties the owner's words with the restatement picks the wrong one | `place`; the restated place forbidden |
| `restated-stale` | recall | a monthly review's venue, changed once by the owner | the assistant restated the old venue in two recaps before the change; the new venue is said once | `place`; the old venue forbidden |
| `buried-preference` | recall | a standing preference (durations in minutes), stated early | the duration comes many sessions later, in hours, beside other sessions' durations | `duration` (minutes), `unit` |
| `spread-total` | recall | boxes of one contact's files, stated in three sessions | the same contact's other files and another contact's same files | `boxes` (the sum) |
| `near-miss` | abstain | none: the item comes up in three sessions, never with a place | other things are placed in those sessions, and the item's review has a venue | `place: unknown`; those places and the venue forbidden |
| `retracted` | abstain | none: a place stated, then withdrawn with no replacement | the stale place | `place: unknown`; the stale place forbidden |

Every probe that asks for a name offers "unknown" in the same words, so the wording never tells a
recall type from an abstention: the place probes of `buried-fact`, `near-miss` and `retracted` are
the same two sentences.

## 3. Instrument settings (identical in every arm)

| Setting | Value | Where it is frozen |
|---|---|---|
| Tool loading, body clock, environment note | as `mind-memory-1`: `tool_loading: eager`, `message_timestamps: gateway`, `environment_note: messaging` | `comparison.*` |
| Clock start | every episode starts at the next 12:00 UTC (`clock_start: '12:00'`); the night crossed before the probe is the only one in the episode | `comparison.clock_start`, protocol `paired-clock-start-1` |
| Seeded history | imported into Hermes `state.db` in every arm and into the ColonyMind ledger in plugin arms before the first turn, without model calls; the base arm reaches it through `session_search`, plugin arms through automatic recall and the memory tools; no arm sees it in its prompt unasked. With `paired plan --history-mode replay` a plugin arm instead replays every pair through its own turn route on the history's dates (claims, capture and vectors form as from live turns; the clock moves back once before the arm's state exists and runs again from the episode's start), and every case's deadline grows by 20 s per replayed owner turn | scenario `history`, protocol `paired-history-1`; replay: `comparison.history`, protocol `paired-history-replay-1` |
| Embedding endpoint | the plan should carry one (`paired plan --embedding-config`), the same in every arm: without it every plugin arm recalls lexically and `full-semantic_recall` cannot differ from `full`. With it a plugin arm waits, at most 300 s, until the history is embedded before the first turn | `comparison.embedding`, protocol `paired-embedding-2` (the claim worker's prior records also search it) |
| Reranker | optional (`paired plan --rerank-config`: endpoint base, model, prompt style, revision, the calibrated cutoff and the fingerprint its calibration recorded), the same in every arm; every plugin arm's host serves it, and the plan records that fingerprint once the served configuration gives it (a config without one, or with one that differs, is refused: recall would switch the cutoff off, and a fingerprint computed from the configuration would pass a cutoff never recalibrated). The evidence's `reranking` says whether the served configuration verified it | `comparison.reranker`, protocol `paired-rerank-1` |
| Process restart | right before the probe, in every scenario; `lifecycle:*` checks join the scenario's checks | scenario `workflow`, protocol `paired-workflow-runtime-1` |
| Iteration and output budget | 8 iterations per turn, 4,096 output tokens, 5 s settle per turn, **900 s** deadline per episode (`GENERATED_DEADLINE_SECONDS`: the history import, the embedding wait and the night) | case inputs |
| Toolsets | common: `file`, `memory`, `session_search`, `todo`; plugin arms add the memory tools and `colonymind_self` | worker, `comparison.plugin_tools` |
| Temperature | provider default (recorded by the plan) | `comparison.temperature` |
| Image | one digest-pinned benchmark image for every arm; its worker declares the current arm-profile protocol (`paired-arm-profiles-10` since the meaning ablations joined; the release image of the first gate declared `paired-arm-profiles-9`, and the development image that first carried the blocks and learning arms its own `paired-arm-profiles-8`) and `history_protocol` | `recipe.container.image_id` |

## 4. Arms

Every arm is a built-in profile of the harness (`paired.PROFILES`; no `--profiles` file).

| Arm | Profile | Role |
|---|---|---|
| `base_hermes` | plugin off | **comparator** for the memory claim; instrument control |
| `full` | plugin on, the `full` switch: every faculty flag and drive weight at its release-candidate value, the resident blocks on | treatment |
| `full-blocks` | `full` with `context.blocks: false` (the `minus_blocks` context switch): the turn carries what the per-turn selection kept, without the block fit | flag arm for `context.blocks` |
| `full-plus-learning` | `full` with `mind.faculties.learning: true` (the mind's lessons and night read the flag; the claim projection reads `COLONYMIND_LEARNING`, which the worker derives from it) | guard arm for turning learning on |
| `full-semantic_recall` | `full` with `mind.faculties.semantic_recall: false` | descriptive, only in a plan with an embedding endpoint |

Arm order rotates by episode; each arm runs in its own fresh container against the same frozen
episode and the same seeded history.

## 5. Primary metric and rules

- **Primary metric: scenario pass** (every check true), graded by `paired_cases.assess` with no
  LLM judge: `answer.json` has exactly the expected keys, every expected value matches
  (`label_one_of`, case-insensitive, or `number`), nothing `forbidden` appears anywhere in the file,
  and every `lifecycle:*` check holds (fresh process, state preserved). `near-miss` and `retracted`
  pass only with `unknown`.
- **Instrument acceptance, read on the same run before any contrast.** A run that fails one is an
  instrument fault, not a gate result:
  - every arm-episode records the seeded history (`tool_evidence.history.sessions` equal to the
    scenario's), every plugin arm's ledger import completes, and with an embedding endpoint the
    vector drain leaves no job;
  - `base_hermes` completes at least 95% of setup turns, and every scenario shows
    `lifecycle:all_phases_completed`.
- **Headroom, read on the dev pilot before any gate.** `base_hermes` passes at most 70% of the
  `recall` scenarios. A family above that is saturated for stock Hermes: the templates are revised
  (a new dataset, a new pilot), never gated as they are.
- **Rules:**
  - Memory: `full` vs `base_hermes` demonstrated (`sign_exact`, alpha 0.05 two-sided over non-tied
    scenarios, at least 6 winning scenarios, cluster-bootstrap 95% lower bound above zero; unit =
    scenario), **and** on the `abstain` group `full` is non-inferior to `base_hermes` (point estimate
    at least -10 pp): better recall must not come from answering what was never said.
  - Memory blocks: `context.blocks` stays on if `full` vs `full-blocks` is demonstrated or
    non-inferior (point estimate at least -10 pp). Otherwise it ships off, labelled "costs recall
    on long histories (delta X pp, CI, n = Y)", until a fix is measured on fresh instances.
  - Learning: `full-plus-learning` vs `full` non-inferior (point estimate at least -10 pp) is a
    precondition for turning `mind.faculties.learning` on, never sufficient: the gain that turns it
    on is measured where claims form from live turns.
  - Anything else is "not demonstrated" (with n, delta, CI and MDE) or "underpowered".
- **Secondary (descriptive, never gated):** per-template pass; attribution errors (files naming the
  assistant's restated place in `assistant-misquote`); stale answers (files naming a forbidden
  value in `buried-correction`, `restated-stale`, `channel-switch`, `retracted`); abstention on
  recall types (`unknown` where a fact was stated); recall packet size (context-route bytes) and
  tokens per turn; history import and embedding wait times; model calls and tokens per arm-episode.

## 6. Pilot and sizing

Placeholders until the pilot. The pilot runs the dev split (seed 7, 27 episodes) with the arms
`base_hermes`, `full`, `full-blocks` and `full-plus-learning` (plus `full-semantic_recall` when the
plan carries an embedding endpoint), 1 repetition, with the release candidate's image. It checks the
instrument acceptance and the headroom rule of section 5; `n` is then the smallest of {40, 60, 80}
with at least 80% power for a +20 pp effect from the contrast's own wins and losses
(`paired_statistics.pilot_sizing`); if even 80 gives under 50% power, the family runs at 40 and a
failure is reported as "underpowered". Until then `n = 40`.

| Contrast | n (pilot units) | wins | losses | disagreement | power at 40 / 60 / 80 | chosen n |
|---|---|---|---|---|---|---|
| `full` vs `base_hermes` | placeholder | | | | | |
| `full` vs `full-blocks` | placeholder | | | | | |
| `full-plus-learning` vs `full` | placeholder | | | | | |

## 7. Relation to mind-memory-1

This family does not replace `mind-memory-1`. The first family measures facts from the episode's
own turns (the claim path: a fact the live turn states is extracted, stored and recalled); this
one measures recall over a long history (the source path: ledger search, reranking, the per-turn
selection and the block fit). A release claim on memory names both results; a regression on either
is reported.

## 8. Known limitations recorded with the plan

1. **The history is sources, not claims.** The history importer retains imported sessions as
   sources only (`hermes_history`, `learning: source_only`): no claim is extracted from them. So
   the learning rules, which fold restated claims into witnesses, have nothing to fold here, and
   `full-plus-learning` is expected to tie `full`; only the guard is read. What the plugin arms
   bring to these scenarios is recall over sources.
2. **The night sees little of the history.** Every scenario crosses one night before the probe,
   so the nightly consolidation runs in the mind arms; its per-contact digests read the last seven
   days and its episode summaries are bounded per night, so months-old history is mostly outside
   them. `full-consolidation` is not an arm of this family.
3. **Channels are sessions.** The history importer records every seeded session on one platform (a
   direct message on the capture platform), and live owner turns are `cli` turns. A channel here is
   a session with its own id and its own style (terse lower-case text messages, formal e-mail), and
   "asked in another channel" is a probe in a new session, in other words.
4. **Old history.** The history is dated from a fixed day in 2026; the probe arrives on the run's
   date, months later, in every arm alike. Every fact is one that does not go stale by itself;
   recall that weights recency treats the whole history as old.
5. **Who said it.** `assistant-misquote` takes the owner's words as the record: the probe asks
   where the owner said the thing goes, and the assistant's restatement, however often repeated,
   is not the owner's statement. A reply that reports both values fails on the forbidden one.
6. **Search is the base arm's only route.** Stock Hermes reaches the history only by searching
   sessions; the memory tool never saw it, because no turn was replayed. This is the product
   difference being measured, not a handicap of the instrument: the plugin arms see the same
   history, and nothing is preloaded for them.
7. **Numbers and labels.** `buried-preference` and `spread-total` ask for numbers and offer no
   "unknown"; a number written as a string fails `number` in every arm alike.

## 9. Plan files and running the gate

```yaml
hypothesis: "full beats base_hermes on recall over a long history, with no loss on abstention"
family: mind-memory-2                # dataset id; content hash frozen by --dataset-dir
split: heldout
seeds: [<fresh 32-bit seed chosen at plan time>]
per_template: <so that the split holds at least n scenarios>
arms: [base_hermes, full, full-blocks, full-plus-learning]   # built-in profiles
reference_arm: base_hermes
contrasts: [[full, base_hermes], [full, full-blocks], [full-plus-learning, full]]
primary_metric: scenario_pass
rule: {superiority: {test: sign_exact, alpha: 0.05, min_wins: 6, ci: cluster_bootstrap_95, unit: scenario},
       abstain_guard: {group: abstain, non_inferior_pp: -10}}
flag_rules: {context.blocks: {demonstrated_or: {non_inferior_pp: -10}},
             mind.faculties.learning: {precondition: {non_inferior_pp: -10}}}
headroom: {arm: base_hermes, group: recall, max_pass: 0.70, read_on: dev_pilot}
pilot: {n: 27, wins: <pilot>, losses: <pilot>, disagreement: <pilot>, power_at_n: <pilot>, placeholder: true}
secondary: [per_template_pass, attribution_errors, stale_answers, recall_abstentions, recall_packet_size,
            tokens_per_turn, history_import_and_embedding_wait, calls_and_tokens_per_arm_episode]
n: 40                                # placeholder until the pilot
repetitions: 1
temperature: provider_default
tool_loading: eager
message_timestamps: gateway
environment_note: messaging
embedding: <the plan's endpoint, the same in every arm>
deadline_seconds: 900
model_recipe: <config hash, recorded by the plan>
image: <digest, recorded by the plan>
```

```sh
export COLONYMIND_HELDOUT_TEMPLATES=/path/outside/the/repository/heldout_memory_hard.py
python benchmarks/paired/generators/generate.py --family memory_hard --split heldout \
  --seed <fresh seed> --per-template <k> --output /private/families/memory-hard-heldout-<seed>
colonymind models paired plan --dataset-dir /private/families/memory-hard-heldout-<seed> \
  --arms base_hermes,full,full-blocks,full-plus-learning --reference-arm base_hermes \
  --repetitions 1 --native-config <private> --native-binding candidate --comparison-policy <private> \
  --embedding-config <private> --container-image sha256:<gate image> --docker-host <forwarded socket> \
  --label memory-hard-gate-<seed> --output <private results>/memory-hard-gate-<seed>
colonymind models paired run ... --output <private results>/memory-hard-gate-<seed>
colonymind models paired report --output <private results>/memory-hard-gate-<seed> --json
```

The dev pilot is the same with `--split dev --seed 7 --per-template 3` and the pilot's arms.
