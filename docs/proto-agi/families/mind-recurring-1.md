# mind-recurring-1: recurring-failure campaigns

Status: pre-registration draft for the recurring-failure family, written on the dev side with the family, the
loader's checks of its probe metadata and the report's recurrence rows (1.13). No pilot has run. Everything a pilot
decides is marked `<pilot>`, and every number this plan chose is marked as an assumption; the bars of section 5 are
left open for the maintainers to set.

## 1. Question

`mind-improve-1` shows whether lessons help on probes that follow training within days and share its wording. It
does not separate three things this family does:

1. **Delay.** Does one correction still help 3, 7 and 14 days later?
2. **Wording.** Does it help when the new request is worded differently from the correction (the paraphrase axis
   lesson selection by meaning targets), and on the procedure's other branch, which the failure never showed?
3. **Later learning.** Does it survive a second, unrelated correction taught in between?

Hypothesis: Hermes with the ColonyMind mind and the lessons faculty on (`full`) passes more of the reworded probes
than the same mind with lessons off (`full-lessons`), at every delay and after the interfering correction, without
losing the same-wording probes.

## 2. Family

| Item | Value |
|---|---|
| Dataset id / version | `mind-recurring-1` (generator protocol `paired-generator-1`, campaign protocol `paired-campaign-1`) |
| Dev templates | `benchmarks/paired/generators/recurring.py`: 6 campaign designs in 3 classes: `procedure` (`dispatch-code`, `crate-count`), `retrieval` (`dock-lookup`, `member-discount`), `tool-misuse` (`case-file`, `settings-edit`) |
| Dev split | seed 7, `--per-template 1`: 6 campaigns, 66 probes (54 warranted, 6 scope controls, 6 contact-rule controls), 12 training artifacts (6 target corrections, 6 interfering ones); the loader content hash is pinned in the generators README |
| Held-out templates | a Python module outside the repository, written by someone other than the faculty's author, declaring the same `FAMILY`, the same design interface (`title`, `when`, `rule`, `same`, six `paraphrases`, `instance`) and at least 2 designs per class over procedures and domains not in the dev module; the generator refuses a path inside the repository |
| Campaign shape | one scenario = one campaign = one container per arm, 15 days in one shared state, one tick ending each day (`advance_clock: 86400`, then `tick: 1`) |

**Days.**

| Day | What happens |
|---|---|
| 1 | The failure: the owner asks for a result (session `day-01`), then corrects it once. The correction states the result and the procedure in full: its title and when it applies, verbatim (`The procedure for <title>, used when <when>: <rule>`), the rule with its other branch, and its scope. Nothing restates it later. |
| 2 | A contact's inbound message asks for a rule the owner never gave (session `contact-1`). |
| 4, 8, 15 | Probes, 3, 7 and 14 days after the correction (`DELAYS`, an assumption: short, medium and long retention inside a campaign whose deadline stays under the cap). Every probe is its own session (`day-NN-k`), in a drawn order within the day, so no probe sees another in its conversation. |
| 6 or 11 | The interfering correction: another design's failure and correction, from another class, in session `day-NN` (`INTERFERENCE_DAYS`, drawn per campaign, an assumption: before the 7-day probes or after them, so the 7-day probes come before the later learning in some campaigns and after it in others). |
| other days | No turn: the clock advances and the body ticks (the night's lesson stage and consolidation run at every tick). |

**Probes.** At each delay, three warranted probes:

| Relation | Wording | Case |
|---|---|---|
| `same_wording` | the failure's own wording, with a new instance | the case the failure showed |
| `paraphrase` | a rewording: at most 1 content term shared with the correction's title and when-to-use (`PARAPHRASE_SHARED_MAX`, an assumption), counted with the lesson matcher's own term function (`colonymind.mind.lessons.terms`) | the case the failure showed |
| `new_instance` | a rewording, as for `paraphrase` | the procedure's other branch (an express order, fragile items, a temporary dock, a large basket, a late order, a missing key) |

A same-wording request shares more than half of those terms. Each design has six rewordings; a campaign draws a
different one for each of its six reworded probes, so they differ from the correction and from each other. Every
probe is an unseen instance: its order number, count, supplier, amount, case number or config never appeared in an
earlier turn.

Two controls, as in `mind-improve-1`, both worded as the failure was, so a lesson found by its words reaches them:

- `scope`: an out-of-scope instance whose right result is the procedure's own abstention (`none`, `unlisted`, `0`,
  a locked config left as it was);
- `contact_rule`: an in-scope instance for the contact who asked on day 2 for another rule; the right result is the
  owner's, and where the tempting value is a distinct token it is `forbidden`.

**Probe metadata.** Every artifact spec carries `probe`: `day`, `kind` (`training`, `warranted`, `control`, with
`control` naming `scope` or `contact_rule`); every warranted and control probe also carries `delay` (days after the
correction), `relation`, `after_interference` and its own `session`; every training artifact names its
`correction` (`target` once, `interfering` at most once and later) and the owner `session` that teaches it (the
request, then the correction). The loader refuses metadata that does not match the campaign's own turns: a
correction whose session does not hold its request and its correction on the correction's day, a probe off the
target correction's day plus its delay, a session that is not an owner session of that day or is shared,
`after_interference` that disagrees with the day the interfering correction's turns fall on, or a partial
declaration (`paired_cases.validate_probes`). Labels that agree with each other but not with the episodes are
refused. Whether a campaign is recurring follows from its family, by dataset id (`paired_cases.RECURRING_FAMILIES`),
not from what its probes declare: a campaign of this family that declares none of the metadata is refused, in one
campaign or in all of them (it would take no snapshots, leave the recurrence rows and be graded on its last day's
files), and so is another family's campaign that declares any of it. A delay is days of the campaign's clock, not
ticks alone: the loader refuses a recurring campaign whose days do not each end with one `advance_clock` of 86,400
seconds and then one `tick: 1`, the last day too, or that advances the clock or ticks anywhere else
(`paired_cases.RECURRING_DAY_END`), so a shortened, missing or extra advance cannot relabel seconds as days.

## 3. Instrument settings (identical in every arm)

The settings of `mind-improve-1` section 3, unchanged: eager tool loading, the gateway clock on every turn, the
messaging environment note, the pinned 12:00 clock start, 8 iterations and 4,096 output tokens per turn, quiet hours
off, and `skill_tools: read` (`skills_list` and `skill_view` for every arm, `skill_manage` for none;
`paired_cases.GENERATED_SKILL_TOOLS`). A campaign case gets 600 s plus 720 s per day, 11,400 s for fifteen days,
under the 4 h campaign cap, and the 8 MiB campaign output bound. The worker keeps each snapshot as what differs
from the final workspace (`campaign_snapshots`), and a plan refuses an image whose worker does not declare the
campaign snapshots protocol (`paired_worker.CAMPAIGN_SNAPSHOTS_PROTOCOL`).

## 4. Arms

| Arm | Profile | Role |
|---|---|---|
| `full-lessons` | `full` with `mind.faculties.lessons: false` | comparator (`--reference-arm full-lessons`) |
| `full` | the shipped defaults, lessons on | treatment |
| `full-plasticity` | `full` with the plasticity faculty on | treatment, once the plasticity wiring lands; not a profile yet |

Arm order rotates by campaign; each arm runs the whole campaign in its own fresh container.

## 5. Metrics and bars

**Primary metric: probe pass**, graded by the artifact checks (`paired_cases.assess`) on the workspace as the
result's own session left it; no judge, no credit for reply text. The case asks the worker for a workspace snapshot
after the last turn of each result's session (a probe's one turn, a correction's verdict; `campaign.snapshot_after`)
and the oracle names the snapshot each result is graded on (`graded_after`). A result repaired on a later day earns
nothing at its delay, one spoiled on a later day keeps its pass, and a snapshot the worker never took grades as a
missing file. Work a session leaves to a later tick does not count for its probe, in every arm alike (the builder's
assumption: the probe asks for the result in that session). The report's `forbidden` hits read the same file. The unit is the probe (warranted and control), repetitions averaged; the cluster is
the campaign, whose probes share one correction. The superiority rule is the harness's (`sign_exact`, alpha 0.05
two-sided, at least 6 winning probes, a campaign-cluster bootstrap interval above zero).

**Recurrence rows** (`report.campaign.recurrence`, descriptive, never gated): per arm, the warranted probes' pass
rate by `delay`, by `relation`, `before` or `after` the interfering correction, and before or after it within each
delay (`delay_interference`: `3:before`, `7:before`, `7:after`, `14:after`), each with a 95% interval that resamples
whole campaigns; per contrast, the treatment minus the comparator in each group, with wins, losses and the same
campaign interval. The interfering correction comes after every 3-day probe and before every 14-day probe, so the
plain `before` and `after` rows also differ in delay: a gap between them can be forgetting over time alone. The
7-day probes fall on both sides, drawn per campaign, so `7:before` against `7:after` is the like-for-like reading
of later learning.

**Lesson diagnostics** (`report.campaign.lessons`): as in `mind-improve-1`, with two additions. A probe is credited
with a lesson only when the lesson was used in the probe's own session, not elsewhere on its day. When the recorded
uses say how the lesson matched the work (the `match` field lesson selection by meaning records, such as
`own_class` or `meaning`), `by_match` splits uses, wins and the probes that had the lesson by that kind.

The family declares no old-family probe, so its report has no old-family row.

**Bars.** Left open for the maintainers. Suggestions, each the builder's assumption and none a rule:

- `full` vs `full-lessons` demonstrated on all probes under the harness rule;
- reworded probes (`paraphrase` and `new_instance`) at 14 days: `full` at least 20 pp over `full-lessons`, with the
  campaign interval's lower bound above 0;
- retention under later learning: `full`'s `7:after` row within 10 pp of its `7:before` row (the plain `after` and
  `before` rows mix delays);
- `forbidden` hits in `full`: 0.

## 6. Pilot and sizing

`<pilot>`: not run. A dev smoke, one campaign per arm end to end on a small local model, shows the harness runs;
it is not a claim. The pilot runs the dev split (6 campaigns, 66 probes) with `full-lessons` and `full` at 1
repetition on a campaign endpoint no live deployment uses. The instrument is accepted on the checks of
`mind-improve-1` section 6, with their thresholds: every arm completes at least 95% of declared entries and all 15
ticks of every campaign, the day-1 training artifact passes in at least 80% of campaigns in every arm, and the
`full-lessons` attempts show no lesson admitted. Sizing follows
`paired_statistics.pilot_sizing` over probes (+20 pp, 80% power); `--per-template 2` renders 12 campaigns and 132
probes.

| Pilot | Branch commit | full-lessons probes | full probes | wins / losses / ties | n | Accepted |
|---|---|---|---|---|---|---|
| `<pilot>` | | | | | `<pilot>` | |

## 7. Known limitations

- Probes of one day run in sessions one after another in one state: a later probe of the day can recall an
  earlier one from memory. The order is drawn per day, so no relation always goes first; the report does not
  correct for it.
- The paraphrase bound counts terms of the title and when-to-use the correction states. The lesson the night writes
  from it may name them differently; the per-use match kind shows what selection actually did.
- A reworded request must still name the same task: a `case-file` probe calls the case a complaint or a ticket, and
  the agent must see that it is one. Same-wording probes are the contrast.
- The interfering correction is unrelated by design (another class). It measures retention under later learning,
  not interference between similar procedures.
- Only the 7-day probes fall on both sides of the interfering correction, about half the campaigns each way. The
  like-for-like rows therefore hold about a third of the warranted probes, and in the dev split three campaigns
  per side; their interval is wide.
- Verdicts are scripted and the correction states the procedure whatever the agent wrote, as in `mind-improve-1`.
- Memory is on in both arms, so the contrast is the lessons faculty's margin over memory alone.

## 8. Plan file

```yaml
hypothesis: "full beats full-lessons on reworded probes at every delay and after later learning"
family: mind-recurring-1
split: heldout
seeds: [<fresh 32-bit seed chosen at plan time>]
per_template: 1                      # 2 if the pilot sizes n above 66
arms: [full-lessons, full]
reference_arm: full-lessons
primary_metric: probe_pass
rule: {superiority: {test: sign_exact, alpha: 0.05, min_wins: 6, ci: cluster_bootstrap_95, unit: probe, cluster: campaign}}
secondary: [recurrence_by_delay, recurrence_by_relation, recurrence_by_interference, recurrence_by_interference_within_delay, lesson_use_by_match, forbidden_hits]
bars: <set by the maintainers>
repetitions: 1
```

## 9. Running

```sh
python benchmarks/paired/generators/generate.py --family recurring --split dev \
  --seed 7 --per-template 1 --output /private/families/recurring-dev-7
colonymind models paired plan --dataset-dir /private/families/recurring-dev-7 \
  --arms full-lessons,full --reference-arm full-lessons --repetitions 1 \
  --native-config <private> --native-binding candidate --comparison-policy <private> \
  --container-image sha256:<image> --docker-host <socket> \
  --label recurring-dev-7 --output <private results>/recurring-dev-7
colonymind models paired run ... --output <private results>/recurring-dev-7
colonymind models paired report --output <private results>/recurring-dev-7 --json
```

A smoke adds `--case-ids <one campaign id>`.
