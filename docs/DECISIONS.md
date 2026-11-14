# Decisions: Instinct

Much of what ColonyMind decides is a short typed decision over a few words: what the owner's reply to an
outreach says, whether a contact opted out, whether a new item is one the person wants no reminders about,
whether an interest is settled, whether an owner message judges the agent's work, whose instruction a stop
beside someone else's words is, whether a report found anything, what the owner's answer to a question keeps,
whether a message to the owner can wait. Instinct (`colonymind.instinct`) is the one mechanism that answers them.

Each such question is a **decision point**: a typed question with a closed set of labels, each with a
description, and one **effect** from a closed set:

- `route`: pick which path handles something;
- `rank`: order what the existing path already admitted;
- `filter`: hold, drop, withdraw or abstain; it only ever removes or delays;
- `ask`: put one plain question to the owner;
- `read`: label the owner's own words.

A point's answer can only select among what its own code offers. It never approves, grants or widens anything,
never lifts a hold, never resumes check-ins the owner stopped, and never retracts or supersedes a stored claim
from a model's reading: the registry refuses a point whose model tier could settle such a label (`approve`,
`grant`, `widen`, `resume`, `retract`, `supersede`, and per point `resume` on `owner_instruction`, `first`,
`second` and `neither` on `question_answer`, `send_now` on `owner_send`). The modules that authorize anything
(authority, the guard and its floor patterns, the session map, the secret stores, the credential and link
resolution, and the answer reader that settles an ask, `mind/answers.py`) never import Instinct, and a test walks
their imports to keep it so. No point's tier settles a yes to an ask: the owner's own message is bound to it by
structure and its words are read by fixed tables ([MIND.md](MIND.md), Asks); `ask_answer` (below) reads an unclear
answer, and a model's yes there asks the owner back until that kind of ask has earned it.

## The tiers

A point is answered by tiers, in order; the first that settles decides.

1. **rules**, in code: a phrase table, the owner gate's prior, a structural check. Final where they speak: then
   no model is asked, live or shadow. Where a point's reader lives with its caller (the reactions phrases, the
   opt-out phrases, the capture call), the caller asks only where that reader leaves the case open.
2. **fast**: a non-generative decision model on your own network (`local-decision.v1`, below), for literal
   yes/no questions and for points measured on it.
3. **reader**: an LLM you already serve, read as option probabilities (added with its measurement).
4. **deliberate**: the mind's own typed call on its router (a JSON enum or `unsure`, 40 output tokens, its input
   cut at 2,000 characters, the day's token budget). For `owner_instruction`, `report_substance`,
   `question_answer`, `mind_wish_scope` and the gate's second opinion this is the **existing path**: the call
   the point made before Instinct, moved unchanged, asked exactly where it was asked before. It answers a label,
   not a probability, and is never a label of record.
5. **fallback**: the point's conservative answer (for most points, nothing: the caller keeps what its reader
   decided).

A fast or reader tier settles only when its calibrated probability clears the label's bar and the label is one
it may settle, and the case is not high stakes (a night-time send, a critical alert on a topic already sent).
Otherwise its answer was **unsure**, and the point goes on to the next tier. A state longer than a tier reads is
never cut: that tier is skipped. A caller applies a label only when the decision says it was applied.

## States: shadow, live, demoted

Each model tier of each point has a state:

- `shadow`: asked where it is eligible, beside the deciding path, and logged; never applied, and never on the
  answer's time (at most 32 shadow asks run at once);
- `live`: its settled answers apply;
- `demoted`: behaves as shadow; set automatically on drift, and live again only by a new promotion.

Every new model tier ships in shadow. A tier goes live on the evidence that it is at least as accurate as the
existing path it falls back to, measured on held-out items. The state is not configuration: it lives in the
instance's `instinct.db` with the evidence for it, and `colonymind instinct status` shows it.

## The points

| Point | Question | Effect | Tiers as shipped | Existing path | What a model tier may change |
|---|---|---|---|---|---|
| `outreach_reply` | how the owner answers an unprompted message: engaged, dig deeper, not interested, not now, stop | read | fast (shadow); no reader yet (below) | the reactions phrase table (`mind/reactions.py`); a reply linked by its topic with no class is engagement | only a topic-linked reply the phrases read nothing in: it takes the model's class; a stop so read pauses outreach as a linked stop does (`reaction.read: decision`) |
| `opt_out` | does a contact ask for no more messages at all | filter | fast (shadow), reader (shadow) | the opt-out phrases (`contacts/optout.py`), and the appraisal's flag as a second net | only words the phrases miss: a yes lowers `may_contact` to `never`, audited as the model's reading without the words; never the owner |
| `no_reminders` | does the person ask not to be reminded about this item | filter | fast (shadow), reader (shadow) | the capture call's HOLD rule for a first mention | only a new item the call gave a due time or a heads-up: a yes creates it held (open, no time, no heads-up); a notice, check-in or cadence for a contact is never asked |
| `interest_settled` | does the owner say their question about this topic is answered or no longer wanted | read | fast (shadow), reader (shadow) | the capture call's complete or cancel on a listed interest | only an open interest the turn names (`outreach.similar`) that the call left open: a yes settles it (`Mind.settle_interest`) |
| `owner_verdict` | does the owner's message judge or correct the agent's earlier reply, rather than ask for something new | filter | fast (live), reader (shadow) | the night's lesson call reports verdicts; the checks admit one that follows an agent reply and is quoted exactly | only an admitted verdict: a no withdraws it (`lesson_verdicts_vetoed`) |
| `owner_instruction` | does the owner, in their own voice, tell the assistant to stop, pause or resume its check-ins | read | reader (shadow), deliberate (existing path) | the reactions reader; a cue beside someone else's words (a quote, a forwarded chat) is put to the typed call | the reader may settle `stop`, `pause_today` or `none`, and the typed call is then not made; no answer: a stop or a pause stands, a resume needs a linked reply. No fast or reader tier may ever settle `resume` |
| `report_substance` | does a research report find anything | route | reader (shadow), deliberate (existing path) | the outreach reader; a null marker beside other words is put to the typed call | `finding` or `empty`; no answer: the digest |
| `question_answer` | what the owner's reply keeps of two statements | read | reader (shadow), deliberate (existing path) | the typed call, once per turn after the question was delivered | the reader may settle `both`, `unclear` or `none`; no fast or reader tier may ever settle `first`, `second` or `neither` (they retract statements) |
| `mind_wish_scope` | does a wish's text ask for more than a yes allows | filter | deliberate (existing path) | the typed call in the night's budget | `within`, `beyond` or `unsure`; no answer: the words are withheld |
| `claim_relation` | does the newer of two statements one person made replace the older, or add to it | filter | fast (shadow), reader (shadow), deliberate (existing path) | the night's typed call before a contradiction is raised ([CONSOLIDATION.md](CONSOLIDATION.md)), charged to the night's budget | only `adds`: the pair is recorded compatible and never asked. A `replaces` (it supersedes a stored claim) is only ever the night's own call's |
| `opinion_none` | does a turn give the agent anything to form or change a view on | route | reader (shadow) | the opinion call, on every turn with admitted evidence | a `no` at 0.99 or more skips the call; never asked for a finding, a request to reconsider or a turn the opt-out phrases flag |
| `appraisal_empty` | does a message carry anything to appraise: a feeling, an outcome, someone's mood, a wish to hear no more, an incident | route | reader (shadow) | the appraisal call, on every turn | a `no` at 0.99 or more skips the call; never asked for revision work (a retained view, an owner's correction, an open incident) or a turn the opt-out phrases flag, since the appraisal is their second net |
| `percept_attend` | whether to ignore an observation, keep a note, look at its original, or tell the owner | route | rules, reader (live, matching calibration required), deliberate (live, after calibrated uncertainty) | retain an unverified concern note | `ignore`, `note`, `look` or `tell`; `look` uses ordinary work authority and reopens the exact original; `tell` passes the owner gate, whose model tiers may only hold |
| `ask_answer` | what the owner's own words answer to one ask: approve, decline, conditional or question, unrelated | read | fast (shadow), reader (shadow) | the answer phrase tables ([MIND.md](MIND.md), Asks); an unclear answer gets one plain question | only an unclear answer to one ordinary ask with a yes or no word first and no hard hedge: `decline` at 0.90 settles no; `conditional_or_question` or `unrelated` asks nothing back; `approve` at 0.97 is never settled by the tier: it asks the owner back (below) |
| `owner_send` | message the owner now, list it in the digest, or drop it | filter | rules (the gate's prior), fast (shadow), reader (shadow), deliberate (the second opinion, existing path) | the owner gate's stages 4 and 5 ([OWNER-GATE.md](OWNER-GATE.md)) | a model tier may only hold: `digest` at 0.80 or `drop` at 0.95 at least, never `send_now`. Once a tier is live it stands in for the second opinion where the prior is unsure: whatever it leaves unsettled (unsure, failing, down, at budget) keeps the prior's choice, and the second opinion is asked only for a high-stakes case (a night-time wake, a critical alert on a topic already sent) |

`percept_attend` is the live senses addition. Its structural rules run before any model: disabled or expired
observations are ignored, and habituation or a missing original keeps a note. An operator-configured per-source
route may then choose a path; a sender's route fields never configure it. Otherwise the calibrated reader can
choose, with the deliberate tier available only after calibrated uncertainty and within the escalation budget.
There is no fast tier or uncalibrated choice. Missing, failed or unmatched model evidence falls back to `note`.
The Instinct log retains each tier's answer as the comparator. Surprise raises curiosity only within its existing
cap, and habituation never closes an open concern. Device descriptions remain `derived_unverified`, never owner
instructions. A `look` worker names the canonical source, its version and the original frame hash; a `tell` remains
subject to ordinary message authority and the owner gate. See [Observations and senses](MIND.md#observations-and-senses).

At most three questions per point per captured turn. Every new model tier ships in shadow. `outreach_reply` has no
reader tier yet: a reply linked to a message by its topic may be a new request rather than an answer, and its five
reactions have no label for that, so a reader would be forced into one of them. Its reader comes with an
out-of-scope label, measured together. The pre-screens' 0.99 is
a code floor, chosen rather than measured: a wrong skip loses an opinion or an appraisal silently, so a screen acts
only where it almost never misses one. The gate's fast tier reads only its short states: with the house facts in
full a state is longer than a decision server reads, and that tier is skipped (never cut).

The reader tier asks the provider `reader:` names, of kind `logits`: its `url` is the OpenAI-compatible base (ending
in `/v1`) of an LLM you already serve, and `model` the name it serves; without either the tier stays off. For
example, `reader: {kind: logits, url: "http://127.0.0.1:8000/v1", model: "your-model", route: chat, thinking: "off",
rotations: 4, timeout_ms: 800}` under `providers`, and `reader: reader`. Quote `"off"` in YAML so it remains a string.
A provider without `timeout_ms` gets its own kind's default: 800 ms for a reader, 250 ms for a decision server.

Each option is shown as `A. <description>`, never its label name. Renaming labels leaves the prompt unchanged.
The reader rotates the options, maps each letter back to its label and averages the distributions. `rotations`
accepts 1 through 4 and defaults to 4 (the existing reader assumption). `route: completions` is the default for
servers without a chat template; it posts the prompt ending at `Answer:` to `/completions`. Its benchmark-only
`echo` mode reads the appended letters' prompt log-probabilities.

`route: chat` requires `thinking: "off"`. It posts to `/chat/completions` with a user turn holding the prompt
without its last line and an assistant turn holding `Answer:`. `continue_final_message: true`,
`add_generation_prompt: false` and `chat_template_kwargs: {enable_thinking: false}` keep the next token in that
answer slot. The request asks for exactly one token, temperature 0, `logprobs: true` and 20 `top_logprobs` (the
existing reader assumption). Echo mode is refused with chat. The chat alternatives are read from
`choices[0].logprobs.content[0].top_logprobs`; spaces and punctuation on a letter count toward its mass.

On both routes, letters must hold at least `LETTER_FLOOR` (0.5, the existing reader assumption) in every rotation.
Missing log-probabilities are `Unavailable`; impossible probabilities are `Malformed`. Chat reasoning, more than
one returned token, or a next token outside the option letters is `NoAnswer`, even if the listed letters hold enough mass. A template that ignores
the thinking switch leaves the point on its existing path. Reading an option after a reasoning block is not built:
replaying that block in a second request would cost a full reasoning pass in the tier intended to save one.

The readout is `names-free/<route>/r<rotations>` and the calibration kind is
`logits/names-free/<route>/r<rotations>`. Records and benchmark answers carry that identity, so a record for the old
prompt, another route or another rotation count cannot settle a reader decision. The benchmark's `decide` and
`check-reader` accept `--reader-route chat --reader-thinking off`; eligibility is read only from the model list and
serving configuration, with no chat or completion probe. Serving requires the letter-mass and per-point gates to
pass, matching calibration records, a logged comparator, automatic rollback on regression and the
`colonymind report instinct` line. Without passing measurements, leave `instinct.reader` empty.

Records measured for a model tier on a served model ship as data (`colonymind/instinct/records/*.json`), each with
the key it was measured on and that key's fingerprint; one whose fingerprint is not its key's, or whose question is
no longer the registered one, is left out. None ships yet: the bake-off below has not been run on a model. Until it
has, every model tier this release adds (each reader tier, and the fast tiers of `claim_relation`, `owner_send` and
`ask_answer`) has no record: it abstains, its raw answers are still logged in shadow, and it cannot be promoted. The
five fast points keep their measured records.

## The owner's answer to an ask: `ask_answer`

The owner answers an ask the way they would answer a person ([MIND.md](MIND.md), Asks): which ask a message answers
is decided by structure, and its words are read by fixed phrase tables. Where the tables read the answer unclear, one
plain question goes back ("Just to check: ...? Yes or no."). `ask_answer` reads such an answer first, and only such
a one: one ordinary ask, bound by a quoted reply or the follow-up (never a list, never a floor ask or one that widens
who gets messaged, who is who or what may run without asking), whose words hold a yes or no word in their first
clause and no hard hedge ("but", "except", "only if", "unless", "instead", a question mark). Its state is the ask's
sentence and the owner's words, within the fast tier's 1,200 characters, never cut. The answer route reads the
message before any turn is captured, so nothing has sealed it yet: every secret in both is hidden before any tier
reads them, in shadow as when live. It asks the question the bake-off measures, so a record fitted there applies
here. It has no deliberate tier: nothing generative is asked while the
owner waits.

What a live tier's reading does:

- `decline` (at 0.90 at least): the ask is settled no, recorded as read by the decision (`answer.reader: decision` in
  the settling history, with the decision's id). A message that carries someone else's words (a quotation, reported
  speech, a framed chat line) is never settled on a model's reading: the plain question goes back. The owner can take
  such a no back: "undo" while it is the latest answer in the chat, within ten minutes, or "I didn't say no" within a
  day. The ask is then open again and goes back to them on its own, where their plain answer settles it. A model's no
  is never applied to an ask whose no does more than record it (a wish's no holds its kind for weeks): the plain
  question goes back instead.
- `conditional_or_question` or `unrelated`: nothing is settled and nothing is asked back; the message goes to the
  model as usual, and the ask stays open.
- `approve` (at 0.97 at least): no model tier may settle a yes. The reading is withheld and asked back instead, as its
  own message: "Doing this: <the ask>, OK?". It is the ask's one plain question of the day. The owner's next plain
  answer binds to it and settles the ask as the tables read it. The tiers withhold `approve`: the refit calibrates it
  and fits its bar like the labels they settle, so a record fitted on labels brings the confirm-back into force; a
  record without its bar never reads a yes.
- anything under its bar, a provider down or without a matching record, or the point in shadow: the tables' plain
  question, exactly as without Instinct.

**Asking back less, by kind of ask.** Once a kind of ask (its kind and type) has 20 right readings of a yes in a row
under the same model, revision and question (a confirm-back the owner then approved, or an approval on the reading
that stood a day with no undo), a model's yes approves that kind without asking back, like any approval: the owner is
told, and "undo" takes it back. So does "I didn't say yes" (or "I never said yes"), within ten minutes as undo words
and past them for a day, the time such an approval takes to stand, for the latest approval in the chat a model read. A
no to the confirm-back, an undo or such a denial is a mistake and starts the count again; so is the owner's own label
on one of those decisions (`colonymind instinct label <decision> <label>`) naming anything but `approve`, which
outranks what the history shows. Only that label does: the history is read as it is now, so an undo spends a
confirmation whatever grade the nightly pass kept from before it. Another model, revision or wording starts the count
at zero. A reading counts only while its decision is in the decision log: one forgotten there (a turn it read erased,
or the owner's answer that graded it: their answer to the question back, their undo; or past the log's retention)
earns nothing and breaks the count, however the ask's history still reads, so a forgotten mistake never joins the
rights around it into approval. Some yeses are always asked back, however many readings were right: anything but the
mind's own task (a message leaves the house; a sealed value's use or a change to who is who is the owner's to give), a
task that holds an effect or a change, and any message carrying someone else's words. Floor and widening asks never
reach the point. The count of 20 and how far back it is read (90 days, at most 500 asks of the kind) are starting
choices.

**Outcomes.** A model's yes is graded from the ask's own history, never from any tier's answer: a confirm-back the
owner answered yes is right, and one they answered no, or approved and then undid, is wrong (explicit); one answer
grades one confirm-back, the latest before it, so a confirm-back the ask was asked again after, unanswered, stays
ungraded. Only the owner's own word grades a confirm-back: their words read by the tables, or their answer at the
console; where a model read their answer to it, the grade is interpreted (never a label, never a count toward
tightening) and neither earns nor spends the count. An approval on the reading they undid (or said they never said yes
to) is wrong (explicit), one that stood a day is right (behavioural). The nightly pass grades these decisions again
while they are in its window and replaces its own earlier grade when the history now reads otherwise (a confirmation
graded right, then undone, is wrong); the owner's own label is kept apart and never replaced. Two explicit wrongs
within seven days tighten the point at once: a wrong yes raises the `approve` bar just above the probabilities it was
read at, and a live tier so tightened is demoted (its yeses are no longer asked back until it is promoted again). A
model's no the owner took back is wrong (explicit); one that stands has nothing that grades it (the ask is closed):
`colonymind instinct label` can. The owner's words are never stored, so these decisions do not re-render.

As shipped both tiers are in shadow with no record: asked beside the tables' path, logged, never applied. A reader's
asks count against the point's hourly budget like any reader ask. `colonymind mind asks-report` counts the answers read
by the decision beside those read by the tables, and the confirm-backs among the plain questions; the bake-off's ask
set reports the false yeses on its adversarial items (reported speech, quoted and injected yeses) even though a yes
only asks back. Neither tier goes live without them: a promotion needs at least 20 adversarial frozen items whose
right label is not `approve` and none of them read as a yes (Promotion and demotion, below).

## The fast tier: `local-decision.v1`

`POST <url>/v1/decide` with a `state` (the short text) and one question named `decision`, always a `choice`.
Its `criteria` are `a`, `b`, `c`, and so on in the point's declared label order, each carrying that label's
description. The instructions are unchanged. Label names stay in the sidecar so the model reads the descriptions
without using a name as evidence: renaming a label cannot change the request or the mapped answer. Yes/no points
also use two letters, mapped back to `yes` and `no` in their declared order.

The answer carries `protocol: local-decision.v1`, `input_truncated: false`, and
`answers.decision: {type: choice, probabilities: {a: ..., b: ...}}`. The provider validates the letters and their
probabilities before mapping them back to the point's labels. Its calibration identity is
`local-decision.v1/letters-r1`. Records measured with label names do not apply and have been removed.

The readout uses one rotation, the declared order, with no rotation setting. The pre-registered rule selected it
because one point passed every gate, versus none with all cyclic rotations; ties would also select one rotation.
Independent questions run concurrently under one deadline (250 ms by default). A timeout or failed request fails
the ask, cancels pending requests and counts toward the tier's breaker.

The tier gives no answer (the point keeps its existing path) when

- no url is configured;
- the state is longer than the model reads: over 1,200 characters (the model's context is 512 tokens, the
  question and its options included) is not sent, never cut to fit;
- the call takes longer than `timeout_ms` (250 ms by default; 1 to 5000 ms), the endpoint is busy (429),
  failing or unreachable;
- the answer is not the typed answer asked for (another protocol, an input the model did not read whole
  (`input_truncated` not `false`), answers to other questions, other labels, a probability out of range, or a
  choice whose probabilities do not sum to one within 0.01);
- no calibration record matches the provider (below), or the calibrated probability is under the label's bar.

The endpoint should be on a private network: the states are the owner's and contacts' words. Nothing in
ColonyMind sends a state to a hosted decision service.

## Calibration records

A record says how one model's raw probabilities become calibrated ones at one point, and the bar each label must
clear there. It is keyed by the point's version (a hash of its instructions, labels and state template), the tier,
the provider kind, the model you configured and the revision the server reports (its answer's `backend`:
`<checkpoint>@<model_revision>`), and it applies only while that key matches: another model, another checkpoint of
it, or a changed question gets no record, and the tier abstains while shadow logging goes on.

There are two methods. Temperature scaling: a probability p becomes p^(1/T), renormalised (for yes/no the log-odds
are divided by T). Isotonic: a non-decreasing step map per label, fitted on labelled answers (below), from the raw
probability of that label to how often the label was right there; a yes/no has one map, for yes, and no is its
complement; a choice's label with no map keeps its raw probability and has no bar, so it never settles. A map is
read only over the raw probabilities it was fitted on: beyond them a label reads its safe bound (0 below the lowest
seen; the last step above the highest, where a yes/no's no reads 0), so it never settles an answer unlike any it saw
by carrying its nearest step outward. A yes/no
answer is yes at or above the yes bar, else no at or above the no bar; a choice's top label counts at or above its
bar. The bar in force is the highest of the point's code floor, the record's bar and a bar you configure:
configuration can only raise it. A record may name the revision it was learned from (its parent): that is lineage
only, never part of the match, so a revision learned from a calibrated one has no record until one is fitted for it.

The shipped letter records cover only `opt_out` under `local-decision.v1/letters-r1`, measured on synthetic
frozen sets with the public typed-decisions checkpoint at revision
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`. The isotonic record was fitted on 71 labels, retaining its map and
support bounds; its yes bar is 0.8 and its no bar is 1.01, so no never settles. Held-out accuracy was 0.907407
against the existing path's 0.814815 on 54 items, renamed flips were zero, reordered flips were 0.088, and the
adversarial gate passed. The record ships as JSON data for the measured checkpoint revision. A server must
report that served revision to use it; a blank or different revision gets no shipped record. Runtime and nightly
maintenance read the same shipped records.

Every other measured fast point fitted no record under this readout, most because the fitted calibration would
never act. They ship no record and their fast tiers abstain with `no_record`, including `owner_verdict`.
The earlier named measurements below are historical results, not records for the letter readout.

## Calibration and learning

Instinct's calibration learns how to read each model's probabilities, from what later happened to its decisions,
and whether a tier has earned the right to act. A configured [plasticity learner](PLASTICITY.md) may separately
train weights; each learned revision still needs its own calibration and preserves Instinct's exclusions.
Everything below runs in the sidecar's
daily task (the nightly pass) and on `colonymind instinct fit`; each threshold marked "my choice" is a starting
value to revisit on evidence, not a measured rule.

**Outcomes.** A resolver reads, after the fact, what the instance's own stores recorded about a decision, read-only;
none is a hook in a caller, and none reads the answer a tier gave: the applied answer is what is graded, never the
grade. Each outcome has a source: `explicit` (your own word or act), `behavioural` (what you did), `synthetic` (a
labelled set) or `interpreted` (what a model made of your reaction). A reaction counts as your word only when the
phrase tables read it from your own words; one a decision model read, an appraisal's dismissal, or a stop whose
speaker the `owner_instruction` point had to settle (a reported "stop") is `interpreted`: kept and shown in the
report as its own teacher source, never a label for the refit and never a count for the tighten rule. The first
resolvers:

| Point | Right | Wrong |
|---|---|---|
| `owner_send` | a message the gate sent, then engaged, welcomed or asked more about (behavioural) | answered with a stop in your own words (explicit), or found not useful (behavioural) |
| `owner_instruction` | a stop still in force a day on (behavioural) | a stop you lifted before the next night's check (behavioural; the stores keep no time of a resume, so "at once" is read as within a day, my choice) |
| `opt_out` | | a model-read opt-out you set back from `never` (explicit) |
| `question_answer` | | an applied answer whose statements come back as a later question on one of its claims (behavioural) |
| `report_substance` | a report read as a finding, sent, then engaged or welcomed (behavioural) | the same, found not useful (behavioural) |
| `owner_verdict` | | a withdrawn verdict you restate in the same session within a day, left standing (behavioural, my choice of window) |

What the stores do not record labels nothing: a message held for the digest and acted on from it, a report read as
empty (it is never shown), a contact who writes again wanting replies, your saying you never said stop. For those,
and for any decision, `colonymind instinct label <decision> <label>` records the right label as your explicit
outcome (judged against what was decided: the applied label, else what the existing path kept). Your label overrides
whatever a resolver recorded for that decision, and your later label replaces your earlier one, in the fit, the
tighten rule and the report alike.

**Erasure.** A decision whose references name a turn an erasure touched is forgotten, with its outcomes: a turn
erased whole, or one left with fewer messages because an erased message's copy was removed from it (the decision may
have read that message). So is one whose grade rests on a later turn so touched: the owner's answer to an ask's
question back, or their undo, whose turns the ask's history keeps (`ask_answer`); the restatement of a withdrawn
verdict (`owner_verdict`); the reaction a sent message or a reported finding was graded by (`owner_send`,
`report_substance`). What it counted stays (the report says how many).

The forget path also unserves plasticity closures trained on those decisions before deleting and rebuilding them.
Calibration records that name a removed revision are withdrawn, leaving the point abstaining until it is refitted.
An interrupted erasure can be retried with the same references.

**The refit.** Per point, model tier and fingerprint, from the labelled answers of the log and of imported sets:

- from 50 explicit or synthetic labels an isotonic map per label the tier reads at a bar (the labels it may settle,
  and those it withholds, such as `ask_answer`'s `approve`), below that a temperature, below 10 nothing (my choices);
- the bars, per label: the most accurate composition with the fallback (the model's label at its bar, else the
  existing path's answer; a withheld label counts as the reading it is), ties to the bar that acts least, "never"
  included, never below the code floor; the accuracy a record carries is five-fold held out (the whole fit redone on
  four folds and read on the fifth);
- behavioural labels only tighten: they can raise a bar, never lower one or shape a map. A deliberate tier's answers
  and a hosted service's are never labels.

Before choosing a label, the refit checks `outcomes.sound` against current sources. A grade resting on a later turn
that was erased or withheld from every projection never trains a refit, even before the nightly forget pass runs.

A refit that would act where the record in force does not, or act otherwise (a first record included), loosens, and
is written only when its held-out accuracy with the fallback is at least the existing path's and the current
record's on the same rows, and the frozen holdout still passes (with the fallback at least as accurate as the
existing path on the frozen items and on their adversarial ones, and, for a tier that withholds a label, reading none
of the adversarial items as it unless it is their right label). With no frozen holdout nothing loosens. A refit is
written at once only when it is shown to settle nothing the record in force does not settle the same way, over every
possible raw answer and not only those at hand: a new temperature or map moves where the bars bite, so higher bars
alone do not show it (for a yes/no this is checked exactly; a choice whose calibration changed always counts as
acting more). Tightenings must also match the current record's accuracy, both its held-out measurement and its
accuracy on the same rows. A refit that would stop all action needs at least the current record's label count;
otherwise it keeps that record, demotes a live tier and logs a warning. A live tier's bar may move at most one
`BARS` step per nightly refit; candidates that move further are skipped.
Two explicit wrongs at a point within seven days tighten its model tiers at once, without waiting for
the night (my choice): the bar of the label they settled (or read at its bar and withheld, which the caller asked
about or approved on) is raised just above the highest of those probabilities, and a live tier so tightened is
demoted. Every candidate also reads planted canaries (raw answers that give a label
no probability, planted with that label); a fit that settles every canary of a label has seen what it must not, so
an audit is opened, shown in the report, and nothing is written. Since a map settles nothing beyond what it was
fitted on, honest labels bunched at one end (all your answers right, at high probabilities) never pass the canaries.
Every record put in force is kept with what it replaced, and so is the one a first local record displaces (the
shipped record, or one carried from 1.11): `colonymind instinct records` lists them and
`colonymind instinct restore <record>` puts one back (a shipped record can be restored by its id at any time).

**Labelled sets.** `colonymind instinct fit --sets FILE...` imports labelled answers of your own provider before the
pass: a JSON document `{"format": "colonymind-instinct-sets/1", "set": name, "hash": the set's hash, "tier": "fast" |
"reader", "provider": {"kind", "model", "revision"}, "flip_rates": {point: rate}, "items": [{"key", "point", "raw",
"gold", "existing", "subset": "fit" | "frozen", "tags", "version", "fields", "group"}]}`. Items that share a `group`
(the renamed or reordered variants of one item) are held out together, never fitted on in one fold and scored in
another. Only a local tier's answers are kept (`local-decision.v1` or `logits`; a hosted service's answers are
refused). They are kept by reference: the raw probabilities, the labels, the tags and a keyed hash of the state
rendered from `fields`, which are then dropped. A `frozen` item is never fitted on, and an item asked in another
wording of its point is left out.

**Promotion and demotion.** `colonymind instinct promote <point> --tier fast|reader [--revision REV]` puts a tier live
on offline evidence, as soon as that evidence is there: nothing waits in shadow for a count of answers, and a live tier
is judged by what it does, with the existing path's answer logged beside it where it is known and demotion automatic.
It refuses without both (and a third for a tier that withholds a label):

1. a calibration record for the fingerprint the tier answers under: the one it last answered under, or, before its
   first answer, the one built from the configured provider and the revision it serves (`--revision`, as the
   provider reports it);
2. an offline pass on the frozen sets imported for it: with the fallback at least as accurate as the existing path
   overall, on the repository's items and on the novel ones, and a measured flip rate under renaming and reordering
   of at most 10%;
3. for a tier that withholds a label (`ask_answer`'s `approve`, which only ever asks the owner back): at least 20
   adversarial frozen items for the fingerprint whose right label is another one (someone else's yes, a quoted or
   injected one), and none of them read as it under the record, whatever the accuracy overall (my choice of 20: about
   two thirds of the bake-off's adversarial ask items; with none wrong in 20 the one-sided 95% bound on the rate is
   about 14%).

Answers the tier already gave under that fingerprint are read again under the record being put live (never what an
earlier record made of them) and kept with the evidence. Two readings of them refuse the promotion: among their
disagreements with an explicit outcome, the existing path right more often than the model; and, over at least 100 of
them, agreement with the existing path under 90% (the drift rule below would demote the tier at once).

The nightly pass demotes a live tier on drift (agreement with the existing path under 90% over its last 100
compared answers; a label the tier withholds, read at its bar, counts as the reading it is, since the caller asks the
owner about it or approves on it, exactly as a promotion's replay and the report count it), on the tighten rule, and
on a fingerprint change: the tier answering under another point version, model or
revision than the one it went live under. A promotion names that fingerprint; a tier shipped live, or carried live
by `colonymind upgrade` from a 1.11 setting, went live under the shipped records and the ones carried with it (a
record fitted here later is not among them); and when none can be established, any answer counts as a change. These
numbers are starting choices, to revisit on evidence. `colonymind instinct demote <point> --tier ...` is yours at any time. Every
change of state is kept with who made it and its evidence (counts, window, record, fingerprint), and the report
shows each point's last one and the window's promotions, carries and demotions. A demoted tier behaves as shadow;
only a new promotion makes it live again.

**Revision lineage.** A model that learns makes a new revision each time it learns, and each revision is a new
fingerprint; without lineage every one would demote the tiers live on it. So a live tier now answering under a
learned revision keeps its state when all of these hold, checked by the nightly pass:

1. the revision has its own calibration record for the fingerprint it answers under (a child is refitted: it never
   reads its parent's record);
2. its record's parent chain (each record names the revision it was learned from) reaches a fingerprint the tier's
   live stretch went live under (its promotion, and every carry since), with a record at every link;
3. it passes the offline pass on the frozen sets imported for its own fingerprint, read now;
4. for a tier that withholds a label, it passes the adversarial pass as well;
5. every revision the chain walks through before it gets there passes 3 and 4 on its own frozen sets too, since
   each would have kept the state only on its own passes.

A server back on a revision of the live stretch (a learner rolled back) is held to the same record and passes, with no
link to walk. The carry waives the shadow evidence a promotion needs; the offline gates and the automatic demotions
(drift, the tighten rule) are the guard. It is kept as its own change of state, live to live, "carried by lineage",
with the child's and the parent's records, each link (and the passes of each one walked through), the fingerprint the
chain reached and both passes as its evidence. Anything else is demoted as a fingerprint change, with the reason
lineage did not carry it. A demotion made while the gates are read, yours or the tighten rule's, stands: the carry
is written by compare-and-set only when the tier is still live with no change of state since the check began. This
includes a demotion between the carry's final read and write: the demotion wins. A tier in shadow or demoted keeps
its state as before. A learned child's nightly promotion calls the carry hook, and carries appear in both the
nightly summary and `colonymind instinct` status output.

**Drift under a learned revision.** A model that improves disagrees with the path it replaces, so for an answer given
under a revision whose record names a parent, a disagreement with the existing path counts against the tier only when
an explicit outcome (your own label, or a resolver's grade from your own words or acts) labels the tier's answer
wrong. One an explicit outcome labels right is measured and not against it. One with no explicit outcome (none,
only a behavioural or interpreted one, or one that says nothing of the tier's answer, as another label being wrong at
a choice point does) is counted as unmeasured, never as an agreement and never against the tier,
and the nightly pass, `fit` and the report say how many. Agreements, and every answer under a revision without a
parent, count as before, and so does a promotion's shadow evidence: this count is drift's, for a tier already live,
and a first promotion keeps the stricter one.

A live tier's settled answer is compared with what the existing path would have answered only where that is known:
at a point whose existing path asks nothing (it keeps the caller's own reading: the five fast points), or where the
existing path was asked beside the tier to compare and its answer logged (outcome `compared`, never applied). Where a
point's existing path asks a model (its deliberate tier is the path: the owner gate's second opinion, the typed
calls), a live settle skips that call, so the runner asks it beside the settle, within the point's hourly budget.
It is the caller's own call on the caller's budget (the night's share, the day's), so it runs inside the caller's
run: off the owner's turn it is made before the settled answer is returned, in turn with the caller's other calls,
so the caller's budget check sees every call before it and nothing is left running, or charged, after the caller is
done (the night's summary included); on the owner's turn the owner never waits for it: it runs in the background,
one at a time per point, within the shadow asks' limit. A settle whose comparison hit the budget (the point's, or
the caller's own), was busy, failed or answered nothing usable has nothing to compare it with: it is counted as
unmeasured, never as agreement, and the nightly pass, `fit` and the report say how many. Being unmeasured is not a
reason to demote.

**Shadow statistics.** `colonymind report instinct` shows, per model tier: coverage (the share of its answers a live
tier would have settled), agreement with the deciding path over the answers that have a comparison (and how many
settled with none), the disagreements and, where an outcome came, whether the model or the existing path was right,
and the same agreement for states seen in the fit sets (imported items and labelled decisions, matched by their
keyed state hash) against new ones; per point its outcomes by verdict and source. For a live tier under a learned
revision it adds drift's own reading, kept apart from the agreement: how many disagreements count against the tier,
the agreement drift reads, and how many disagreed with no outcome to say who was right.

**Re-rendering by reference.** The log keeps references and a keyed hash of each state, never the words. Where a
point's references locate its state, the report says what share of its decisions re-render: the fields found again
in the stores, rendered through the point's template, and matching the logged hash under the instance's secret. The
states found again are the owner's or a contact's turn (`owner_instruction`, `opt_out`, `owner_verdict`, and the
pre-screens `opinion_none` and `appraisal_empty`), the mind's rows (a finding, a message and the reply to it, a
question with the turns before the reply), and for `no_reminders` the item the turn's capture made, which the
decision names by a keyed hash of its words. Five more points read their records: `interest_settled` reads the
interest named by its hashed key beside the captured turn; `mind_wish_scope` reads the wish's ask record;
`ask_answer` reads the captured answer through its turn or answer-message key and checks the hashed contact name
used in the ask's sentence; `claim_relation` reads the claims at their pinned source messages, with keyed hashes
for the topic and local dates; `percept_attend` reads the observation's source and retained event metadata, only
when the source belongs to the owner. An answer whose message record has no link to a captured turn stays missing.
Refused wishes have no ask record to read, and withheld wish words cannot reproduce the earlier state. Rows logged
without the required references keep `no_loader`, with reason `before_references` ("logged before its references"),
and cannot train. A changed or erased source, or a field read at another moment (a
question read after a later turn arrived, an appraisal that also quoted the person's earlier words), does not. Nor
does a turn every projection of the memory leaves out, whether the decision read it as its own or as the reply
before it: an answer that drew on something whose speaker you corrected, or a turn an erasure kept a copy out of.
Every other point says why, by a reason code in the report:

| Reason | Points | Why |
|---|---|---|
| `redacted` | `owner_send` | the gate's house facts and its message text are kept only redacted, and the facts are the gate's view of that moment |
| `external` | `cron_*`, `inbound_triage`, `tool_screen`, every operator point | the words live only in the store of the gate that asked |

Any learning from logged decisions can use only the rows that re-render.

**Training rows by reference.** Learning in weights reads the log through one reader, which yields, per decision
that re-renders and has an outcome that is a label, one row held in memory only: the state text, its keyed hash, the
label, the teacher and the narrowest audience its sources show. The teacher follows the outcome: your own label
(`colonymind instinct label`) is `owner_label`; a resolver's grade from your own words or acts is `verified_outcome`;
both count as proof. A behavioural outcome is `behavioural_proxy`, which only ever pushes away from the label your
behaviour showed wrong, and a behavioural "right" teaches nothing, since a proxy is never a positive. A synthetic one
is `synthetic`, and an interpreted one (a model's reading of your reaction) is no label. Where an outcome names only
the label that was wrong, a yes or no row takes the other label, as the refit does, and a choice row pushes away from
the wrong one. The audience is the owner's where any source is yours or the mind's own record, a contact's where the
narrowest source is a person you filed, a guest's where every source is a sender the agent remembered on its own;
a row whose sources show none is left out. The label comes from the strongest outcome whose grade still stands: a
grade that rests on a later turn now erased, or left out of every projection, teaches nothing, even before the
nightly pass forgets its decision. The state text is never written: no database, file or log line holds it, and an
erased source leaves its rows out by construction.

## When a provider is down

- Each provider has a breaker: three failures or timeouts in a row skip it for five minutes (`provider_down`),
  so a dead endpoint does not cost its time limit on every call.
- A failed, slow or missing provider, or one serving a checkpoint no record matches, leaves the point on its
  existing path, with exactly the model calls that path makes: nothing escalates to the reader or to the mind's
  model because a provider gave no answer, and a later tier is then not asked in shadow either. Only an unsure
  answer escalates, and only past the existing path within the point's budget (20 escalations an hour; every
  reader ask, live or in shadow, counts against it, and at budget the point keeps its existing path), so neither an
  outage nor a flood of crafted words can load the shared endpoint.

## The decision log

`instinct_decisions` in `instinct.db`, in the instance's state directory (the upgrade's backup copies it with every
other store, and its secret `instinct.key` with it, private, so a log restored from that backup keeps matchable
hashes). One row per decision: the event; the point and its version; per tier its state, raw and calibrated
probabilities, latency and outcome, and the calibration identity each answer was read under (the provider kind, the
model you configured, the revision the server reported and the fingerprint of that key, matched by a record or
not); the final label, whether it was applied and the effect; a reason code;
references (turn, intention, question or message ids; for a proposed reminder, its turn, its place in the capture
call's answer and its description; a value that does not look like an id, and a description always, is kept only
as a keyed hash); and a keyed hash of the state as sent, an HMAC under the instance's own secret (`instinct.key`),
so a short state cannot be guessed from its hash. A secret that cannot be read, or a file that holds no whole
32-byte key, never stops a decision: its hashes then take a key of that process alone, which nothing later
matches, and never an empty or cut one (restore the file from a backup, or remove it to make a new one;
`colonymind doctor` fails such a file, `instance-secret`). No
text: no column holds the words a point read. Rows older than 180 days are dropped and their counts kept. The
outcome columns are for the owner's later word on a decision.

## Configuration

```yaml
instinct:
  providers:
    local: {kind: local-decision.v1, url: "", timeout_ms: 250}   # your decision server; empty: no fast tier
  fast: local                         # the provider the fast tier asks
  reader: ""                          # empty: no reader tier
  points:
    owner_send: {bars: {drop: 0.97}}  # bars can only be raised above the code floors
  external: {}
```

`colonymind.yaml` validates the section by name (a known provider kind, an http(s) url or empty, a time limit from 1
to 5000 ms, a known point and label, a bar from 0 to 1) and exports it to the sidecar as `COLONYMIND_INSTINCT`
(JSON); a value already in the process environment wins, so a service unit can pin the whole section.

Upgrading from a release with a `decisions:` section: `colonymind upgrade` carries it once, after it has stopped
the sidecar service and taken its backup. Its `url` and `timeout_ms` become the `local` provider's; a point's
`enabled`, read as before (true or false, or a word or number for one, such as `"off"` or `0`), becomes its fast
tier's state (true: live, false: shadow), and a value the old loader refused stops the upgrade before anything is
written; its `temperature` and `abstain` band become local calibration records for the historical measured
checkpoint and a server that reports no revision, since the setting tuned those measured defaults.
A server that reports another checkpoint gets no record, and the tier abstains until one is fitted for it. The
sidecar refuses to start with a `decisions:` section left, and names `colonymind upgrade`; nothing reads the old
`COLONYMIND_DECISIONS_*` variables, and `colonymind doctor` warns about one still set.

## Seeing what it does

- `colonymind instinct status`: the providers configured (their kind, never their address), each point's tiers
  and states, and the last day's decisions.
- `colonymind instinct points`: every point with its version, type, effect, tiers, states and labels.
- `colonymind instinct log [--point P] [--since 24h] [--limit 20] [--json]`: recent decisions, ids, outcomes,
  labels and references.
- `colonymind report instinct [--since 7d]`: per point and tier its state, how often it answered, abstained,
  failed and was skipped, the escalations and budget hits, its latency (p50 and p95) and the record in force (the
  one its latest answer was read under, or none when that answer matched no record).
- `GET /v1/host/health` carries `instinct`: each provider's breaker and the tier it serves, and per point its
  states and counts since the sidecar started. `colonymind doctor` says whether a fast provider is configured,
  counts the model tiers by state and the last day's decisions.

All of them read ids and counts: none shows the words a point read.

## Measurements (2026-09-26)

The decision model measured is Laya (Apache-2.0; ModernBERT-large, 512-token context, SDK 0.3.4, model revision
`1c5edc17`), both its English root and its typed-decisions checkpoint, served one request at a time on one GPU.
The existing path's model calls ran on the evaluation model (an open-weight chat model quantized to 4 bits) on a
shared endpoint.
Sets: 98 replies, 102 opt-out phrases, 30 reminder turns, 38 interest turns, 77 owner messages (345 in all; of
them 68 are `novel`). Five owner messages are left out of `owner_verdict`: the lesson call gave no answer
that parsed (four) or failed (one), on two tries each (the night would have learned nothing from them either).
This is the second run of the measurement: the first let neither capture point act (its composition treated
every model-call row as one the existing path had answered); the harness now asks the model exactly where the
sidecar would (`measure.eligible`), and both steps were run again.

Typed-decisions checkpoint (the shipped records are this checkpoint's). Accuracy is correct/n,
with the repository's own items in brackets; ECE is the expected calibration error of the top answer;
"with fallback" is the composition the sidecar wires, held-out over five folds; helped / hurt counts the answers
it changed from the existing path's, right and wrong; T and the threshold are fitted on the whole set (1.01 is
"never act").

| Point | Existing path | Model alone, zero-shot / ECE | Model alone, T fitted / ECE | With fallback | Helped / hurt | T, threshold | Live |
|---|---|---|---|---|---|---|---|
| `outreach_reply` | 74/98 (66/68) | 62/98 / 0.25 | 62/98 / 0.11 | 71/98 (60/68; novel 11/30) | 3 / 6 | 0.33, 0.55 | no |
| `opt_out` | 89/102 (79/86) | 83/102 / 0.17 | 83/102 / 0.04 | 89/102 (75/86; novel 14/16) | 5 / 5 | 0.33, 0.85 | no |
| `no_reminders` | 30/30 (22/22) | 24/30 / 0.28 | 24/30 / 0.14 | 30/30 (22/22; novel 8/8) | 0 / 0 | 0.25, never | no |
| `interest_settled` | 37/38 (31/32) | 32/38 / 0.23 | 32/38 / 0.16 | 37/38 (31/32; novel 6/6) | 0 / 0 | 0.25, never | no |
| `owner_verdict` | 65/72 (57/64) | 67/72 / 0.29 | 67/72 / 0.10 | **70/72 (62/64; novel 8/8)** | 5 / 0 | 0.25, 0.03 | **yes** |

English root checkpoint, for comparison: `outreach_reply` 56/98 alone, 70/98 with fallback (hurt 7);
`opt_out` 62/102 alone, 84/102 with fallback (hurt 7); `no_reminders` 26/30 alone; `interest_settled` 35/38
alone; `owner_verdict` 69/72 alone (ECE 0.23, 0.03 with T fitted) but 62/72 with fallback (helped 0, hurt 3).
Neither checkpoint beats the phrase tables at a point they cover; the typed-decisions checkpoint is the one to
serve.

Latency. The decision model's own time was 16 to 18 ms at the median and under 19 ms at p95 for every point
(inputs up to 196 tokens). A round trip on the same private network was 19 ms at the median (20 ms p95); across
two extra network hops it was 37 to 42 ms (p95 73 to 137 ms), inside the 250 ms limit. The phrase tables take
under 0.3 ms. The model calls they would stand beside took 13 s (`interest_settled`), 21 s (`no_reminders`) and
39 s (`owner_verdict`) at the median on the shared endpoint, queueing included, and up to 79 s at p95; the
decision model adds its answer to those paths, it does not replace the call.

What the numbers say:

- `owner_verdict` is live. The lesson call reported five of the owner's plain requests that follow an agent
  reply ("p-95 has a shipment of 52 kg going out. I need the fee.") as verdicts on the work, and missed two
  verdicts. The decision model withdrew all five and no real verdict, held out. It errs only toward learning
  less: a withdrawn verdict admits and scores nothing. It acts only when the calibrated probability of a
  verdict is at most 0.03 (about 0.30 before calibration); the requests measured 0.15 to 0.37 before
  calibration and the verdicts 0.36 and above, so the margin is narrow and the band keeps to the clear side.
- `outreach_reply` and `opt_out` stay in shadow. The phrase tables are right on almost every phrase the repository
  labels, and where they say nothing the model more often invents a class ("I need to focus on the grant report
  this week" read as not now, "Drop it" and "Not interested" read as an opt-out) than finds a missed one. On
  the `novel` paraphrases it helps (opt-out 14/16 against 11/16), which is not enough to make it live.
- `no_reminders` and `interest_settled` stay in shadow. On this run the capture call held all 30 no-reminder turns
  and left one interest open ("that one is theirs and I do not want you on it"), which the model does not
  settle surely enough; so there is nothing for the model to fix. The capture call is not steady across runs:
  19 of its 345 answers changed from the first run, when it gave a reminder to three no-reminder turns (the
  capture test's own sentence, "I am handling it myself. No reminders about it.", among them). Over that first
  recording the corrected harness fits `no_reminders` to act at 0.4 (T 0.25) and fixes two of the three with no
  harm, 29/30 held out, which would make it live; over this one it fits "never". The point stays in shadow by the rule,
  and the case for it rests on how often the capture call misses a hold, which one run does not settle.
- The existing paths' own misses the sets exposed: a contact's "Stop checking in on me", "Only message me when I
  ask" and five like them are an owner's stop cues (`mind/reactions.py`) but not contact opt-out phrases
  (`contacts/optout.py`); and a reply whose positive cue is aimed at a detail the outreach did not name ("Look
  into it further and find out its test method") is engagement, not "dig deeper".

The full results, the held-out errors included, are in `benchmarks/decisions/results/2026-09-26.json`.

## Measuring again

`benchmarks/decisions/` holds the labelled sets and the measurement. The sets are built from the repository's
own words: owner turns of the dev families (`benchmarks/paired/generators`, dev seeds 7, 11 and 13, at most
four per template and label; never a held-out split), the phrases the sidecar suite labels (the reactions
paraphrases and ordinary speech, the opt-out family, close variants and near misses, the capture and lesson
fixtures), and a smaller set of paraphrases written for this measurement in forms neither the phrase tables nor
the templates use (`novel`, reported apart).

```sh
python benchmarks/decisions/measure.py current --model-url <openai-compatible /v1> --model <name> --out current.json
python benchmarks/decisions/measure.py decide --url <decision endpoint> --label <checkpoint> --out answers.json
python benchmarks/decisions/measure.py analyse --current current.json --answers answers.json [...] --out summary.json
```

`current` runs each point's existing path: the phrase tables, or the production prompt (the capture call's
`SYSTEM` and `build_prompt`, the night's `LESSON_SYSTEM` over a one-session packet) on a model endpoint, read the
way the sidecar reads it. `decide` asks the decision model every item's question, one at a time. `analyse`
reports, per point and checkpoint: the existing path's accuracy and time; the model alone, zero-shot and with a
fitted temperature, with its expected calibration error (ten equal-width bins over the top answer's
probability); and the model with the fallback, composed as the sidecar wires it (asked only where the sidecar
asks: the phrase tables said nothing, the capture call gave the item a reminder, it left a named interest open,
or the lesson call reported a verdict), with the temperature (least
negative log-likelihood) and the abstain threshold (most accurate; among equals, the one that lets the model
act least, "never" included) fitted by five-fold cross-validation, so the accuracy reported is the held-out
folds'. A point's fast tier ships live only when that is at least the existing path's accuracy overall and on
the repository's own items alone, and the threshold fitted on the whole set is not "never".

## The bake-off: which tier settles which point

The measurement above covers the five fast points against one decision model. The bake-off asks the wider
question for every point, and for the Hermes-side decisions: where does a local tier settle most decisions at
no loss of held-out accuracy, how far is a decision model from a reader over an LLM the operator already serves,
and how often does each point's ambiguous band fire at all. It runs offline, on synthetic items only; its
results land here and in `benchmarks/decisions/results/` when a campaign has run.

**The prize count first.** Before any set is asked, `prize.py` counts from the paired benchmark's development
records, with no model, how often each typed call's band fired (per 100 owner turns) and how often the opinion,
appraisal and capture calls returned nothing; counts only, never text. Each answer is read by the reader production
uses for its call first: one production would reject is no answer, never an empty result, so a failed call cannot
pass for a call worth skipping. A campaign is read only when its own plan
(`paired.json`) declares a development split; a held-out campaign, whatever its directory is called, another
split, or no readable plan is skipped before any trace is opened, and counted. A point whose band fires less than once
per 100 owner turns leaves the bake-off with that number as the reason; a pre-screen is built only where the
count shows calls worth skipping.

**The sets** (`catalog.py`), labelled by construction and built under a seed from the repository's own words and
templates (the phrases the suite labels, the dev families, the contradiction-key dev pairs, the mind's own
question wording), each with a `novel` subset of at least a quarter (forms neither the tables nor the templates
use) and an `adversarial` subset (reported speech, quoted and injected answers or instructions, near misses,
out-of-scope input):

| Set | Point | Items |
|---|---|---|
| the five measured sets, with new novel and adversarial items | `outreach_reply`, `opt_out`, `no_reminders`, `interest_settled`, `owner_verdict` | 415 |
| asks, each answer paired only with an ask it can answer (an answer naming an action, a price, a draft or two people goes with an ask about it) | `ask_answer` (proposed: approve, decline, conditional or question, unrelated) | 178 |
| the owner gate, in a short state a decision model reads whole and a full one it never gets cut | `owner_send` | 116 |
| the owner's instruction, the answer to a question, whether a report found anything | `owner_instruction`, `question_answer`, `report_substance` | 180 |
| how two statements stand | `claim_relation` (proposed) | 90 |
| cron wake, eight jobs each with its own factual question, asked beside the job's configuration (the order, target price, topics, events or area it watches) | `cron_<job>` (proposed) | 157 |
| inbound triage in direct and group chats | `inbound_triage` (proposed) | 130 |
| tool-result screen, clean and planted | `tool_screen` (proposed) | 190 |

`frozen.json` holds each set's hash, counts and seed; every step refuses a set whose hash differs (a phrase it is
built from changed), and `measure.py freeze --check` names it. Re-freezing is a step taken before a campaign,
never during one. Every arm's answers file records what it answered: each set's hash and, for the arms that pose
the bake-off's own questions (B, C, D, E), each point's question fingerprint (its version and every perturbed
wording). `bakeoff` scores nothing until every file's record is the current one, so answers from an earlier
freeze or wording never meet new labels. Public injection sets for the screen are read from pinned upstream files (`public_sets.py`:
each file's hash and license checked, never committed) and reported apart.

**The questions** of points no unit registers yet carry proposed wording that passes the registry's checks,
exclusions included (a model never settles an ask's `approve` or a relation's `replaces`), so a measured wording
can ship as measured. Every question is also posed with its options shuffled, renamed, with neutral names (`a`,
`b`, `c`) and the descriptions carrying the meaning, with the descriptions of polar labels swapped, and without its
out-of-scope label; each answer is mapped back to the meaning it carried, so a flip says the provider followed a
name or an order rather than the question.

**The arms:**

- **A**, the existing path: the tables, readers and the gate's prior first; the production model call only
  where the path makes one today. The ask reader's tables, and stock Hermes with no gate, ask no model.
- **B**, the deliberate typed call at every item: the production prompt where one exists, else one of the same
  shape from the point's question.
- **C**, the decision model (`local-decision.v1`), one question per call.
- **E**, the reader (`logits`), with its option-order rotations; run only when the endpoint's model list names the
  model and its serving configuration says it returns log-probabilities. `decide --provider logits` reads both
  first (`check-reader` shows the same check alone), never by a chat or completion request; unless both hold, the
  arm is written as dropped with the check's reasons, no completion request is sent, and the summary lists it
  under `dropped_arms`.
- **F**, Instinct's own composition: for each item the caller's reader first, then `Instinct.decide` with the
  recorded answers replayed as its providers (C only on a literal point, then E, then B), so every rule of the
  runner applies unchanged: rules final where they speak, a long state skipped and never cut, an excluded label
  never settled, a high-stakes case escalated, a failure never escalated. Its bars are fitted on the training
  folds as the runner decides each case: a case the rules settle never moves a bar, and a tier is calibrated and
  scored only on the cases it is asked.

The arms answer through the sidecar's own provider clients, so a measured answer is one the sidecar would act on.
No model is loaded for the bake-off: an arm whose model is not already served is dropped and recorded.

**The measures**, per set and arm: held-out accuracy (five folds stratified by label, the variants of one message
kept in one fold: the gate's short and full states, the same words under another ask, the same message from another
chat or sender, every screen result made from one passage, clean or planted; F's fit folds the same way), overall and on the repository, novel and adversarial items; calibration error raw,
with a fitted temperature and with a fitted isotonic map; AUROC; F's coverage (the share settled without the
LLM); latency p50 and p95; flip rates per perturbation; the false approves on the adversarial asks with their
one-sided 95% bound; the false skips of a cron run or an addressed message; the false flags on clean tool
results.

**The criteria** (fixed before a run; `report.CRITERIA`): the local tiers are shown to pay when, on at least two
of the three new mind sets (asks, the gate, the typed enums), F is at least as accurate held out as A and B,
settles at least half the items without the LLM and has a measured fast-tier p95 of 50 ms or less (no measurement
is not a pass); a flip rate over 10%
under renaming or reordering keeps that provider off that point; any false approve on an injected, quoted or
reported yes keeps `approve` off even as a confirm-back; and for the Hermes-side sets, no false skip that hides a
delivery (cron), no skipped addressed message (triage), and at most 1% false flags on clean results (screen).
With 50 to 200 items per point only large gaps show: "no gap shown" is not "equal".

```sh
python benchmarks/decisions/prize.py <paired results> --out prize.json
python benchmarks/decisions/measure.py freeze --check
python benchmarks/decisions/measure.py current --sets <set>... --model-url <openai-compatible /v1> --model <name> --out a.json
python benchmarks/decisions/measure.py deliberate --model-url <openai-compatible /v1> --model <name> --out b.json
python benchmarks/decisions/measure.py decide --provider local --url <decision endpoint> --out c.json
python benchmarks/decisions/measure.py check-reader --url <openai-compatible /v1> --model <name> --serving-config <file>
python benchmarks/decisions/measure.py decide --provider logits --url <openai-compatible /v1> --model <name> --serving-config <file> --out e.json
python benchmarks/decisions/measure.py bakeoff --existing a.json --deliberate b.json --answers c.json e.json --out summary.json
```
