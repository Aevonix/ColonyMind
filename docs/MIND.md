# The mind: drives, concerns, feelings, deliberation, goals and the closed loop

ColonyMind is the mind; Hermes is the body. The mind makes only tool-less
model calls through its own router (at most one per tick). Every effect on
the world is a Hermes kanban task on the `colonymind-act` profile (idempotency
key `mind:<id>`) or a message the plugin sends verbatim. There is no second
executor.

This document describes what ships: the drives, the concerns (the
workspace), the agent's own feelings, deliberation, agent-owned goals, the
tick, authority, asks, the audit log, the outbox, outcomes and their
verification, lessons, skills and the off switch.
The design is in
[docs/proto-agi/PROTO-AGI-ARCHITECTURE.md](proto-agi/PROTO-AGI-ARCHITECTURE.md)
(sections 3, 4.3, 4.5, 4.8, 5 and 7).

## The loop

1. **Capture.** After every person-scoped turn the projection worker runs the
   `commitment_extract` task on the router and records durable commitments
   ("I'll send you the report by 3pm") and owed deliverables in the commitment
   store. It is on whenever a router is configured; duplicates of open items
   and of items rejected as invalid or duplicate are skipped in code, and an
   item withdrawn or dismissed as `obsolete` is shown to the model as closed
   (recorded again only on a clear fresh commitment). Deadlines resolve
   against the turn's own time, so a promise captured late (an outage, a
   backlog) that is already due is imported `overdue` with its original
   deadline, not dropped.
   The prompt lists the person's open items by number, with the person's
   previous turns of the last hour for context, and a later message that
   changes one is an action on it rather than a new row: `reschedule`
   (earlier or later; with no time it is a hold, the deadline is cleared and
   the item stays open; a first mention the person wants no reminder about is
   created held, with no time), `complete` or `cancel` (resolved `done` or
   `obsolete` by `conversation`). On the owner's turns the owner's open
   interests (what curiosity would research) are listed after the items,
   numbered on from them, and a `complete` (answered) or `cancel` (no longer
   wanted) on one settles it in the same pass (`Mind.settle_interest`: its
   level goes to 0, research on it not dispatched yet is cancelled, and its
   research concern is resolved as a stored finding resolves it). Where the call gave a new item a
   reminder or left a named interest open, the decision model may hold the item or settle the interest
   when those points are enabled ([DECISIONS.md](DECISIONS.md): `no_reminders`, `interest_settled`),
   never the other way. A stall or a partial update changes
   nothing; an obligation between other people is not an item; an item
   conditioned on another event, or with no clear time, gets no deadline.
   An action must identify its row: the listed wording, and the listed
   deadline when two rows share it; an empty or merely similar wording is
   ignored, never guessed. The write is a compare-and-set against the
   description and deadline that were listed, so an extraction still
   thinking while the owner corrects the row never overwrites the
   correction; the job reruns once against the fresh state. A new item
   records its counterpart (`metadata.counterpart`, the contact as the
   conversation named them, `owner` for the owner), and a turn attributed to
   that contact lists the item next to their own, so "I need it earlier" or
   "got it, thanks" from the other side reaches it. It also records who owes
   the work (`metadata.obligor`: `owner` for the owner's own promise or
   reminder, `assistant` when the assistant took it on, or the other party
   who promised it); the duty drive reads that to tell a reminder from a
   task. An item whose obligor and counterpart are two different named
   third parties (neither is the owner, by any name the owner goes by, nor
   the assistant) is an obligation between other people and is not
   recorded; a deliverable or a message the assistant sends is never one.
   One person's capture jobs land in order: a later job waits for an
   earlier one still pending, running or backing off. The order never
   becomes a stall: a job backing off after a transport failure holds the
   person's later jobs for at most 15 s at a time, after which the worker
   retries it early and uncharged, so one failed call delays a person's
   captures by seconds; only its scheduled attempts (after 60 s, then 120 s)
   spend its three tries, and a job with nothing queued behind it keeps the
   full backoff.
   An unusable answer (cut off, not JSON) is retried at once, up to three
   times; only transport failures back off. When the person asks for a word
   before the deadline ("give me a heads-up at half three") the item keeps
   the deadline as `due_at` and carries `metadata.heads_up_at` (ISO) or
   `metadata.lead_minutes`; the heads-up is part of that item, never a second
   one. A moved deadline moves an absolute heads-up by the same delta (a hold
   drops it); a turn that states a new warning time replaces it.
   A message to a third party later or on a condition ("if Kim has not sent
   the draft by 3, ask her for it"; "if the venue is not confirmed by 5, tell
   them the booking lapses") is one item due at that time, obligor
   `assistant`, with `metadata.kind` `notice`
   (the owner's own words, sent verbatim) or `check_in` (a topic of at most
   six words, never a figure, amount or code, composed at send time), the
   `recipient` as named and `grant: owner`. It reaches someone the owner did
   not write to, so it exists only on the owner's own words asking for it:
   the extractor quotes them (`asked`, naming the recipient), code checks the
   quote is the owner's and names the recipient (or refers to one named in the
   recent conversation or a listed open item: "if they have not sent them,
   chase them yourself"), and the claim-review pass
   (`source_claims.review_proposals`) confirms against the whole message, with
   that conversation and those items to show whom the words mean, that the
   words ask the assistant itself to contact that person. Anything short
   of that ("tell me / let me know / flag it to me if it lapses", a recipient
   that is the owner, a refused or failed review) is the owner's own reminder
   (`kind: reminder`, obligor `owner`); the row keeps the quote and the
   review's decision, and the duty drive sends only a row that carries them
   (a grant stored before the review existed is the owner's reminder). Only the owner's own turn keeps the grant; the same
   shape from a contact is an ordinary item.
   A word the person asked for (a reminder, a nudge, a word if something has
   not happened) is `kind: reminder`: a message when due, never a task, on any
   lane. It is never merged into another item: beside an item of the same
   turn or an open one it is its own row, whatever its wording, and a
   heads-up is only what the extractor states on the item itself
   (`metadata.heads_up_at`). A new item is a duplicate only of an open item
   that is the same item exactly: the same kind, obligor, counterpart,
   recipient and deadline, and the same description after only Unicode NFC
   normalisation, casefolding and whitespace collapsing. A new item with a
   listed item's own wording and kind and a new time moves it,
   compare-and-set; a merely similar one (the Q4 report beside the Q3 report)
   never does. The
   person's own words for the time are kept as `metadata.due_text` and shown
   beside the converted date in Pending Commitments. A message to
   pass on now ("tell Kim the meeting moved") is the reply's own job: nothing
   is recorded, and capture drops a third-party message due within three
   minutes of the turn, so the mind never sends a second copy of what the
   turn already sent. A deliverable goes only to the turn's own person: one
   the model records for a named third party is read as a notice to that
   party, so the words never go back to whoever asked.
   A recurring check-in the owner sets for a contact ("check on Kim every
   week about the kitchen quote") is one undated item, obligor `assistant`,
   with `metadata.kind` `cadence`, the `recipient` as named, a `topic` under
   the same six-word rule and `cadence_minutes`. Only the owner's own turn
   records one, and it never carries a grant: permission stays the contact's
   `may_contact`.
2. **Tick.** Every 60 s the sidecar runs the timers (ask expiry, deferred
   intentions, expectation resolution, retention, the nightly backup, and,
   with the mind on or off, the nightly vector compaction of
   [EMBEDDING-GENERATIONS.md](EMBEDDING-GENERATIONS.md)), then
   **drains the capture jobs still pending** (`CommitmentExtractor.drain`
   over the same ledger the projection worker uses: claimable jobs are run,
   a job the worker holds is waited for), bounded to 5 s on the timer and
   30 s on a forced tick (`POST /tick`, the CLI, the benchmark), so a promise
   made seconds ago is a row before the drives look; the tick summary's
   `capture_drained` records what landed and how long it waited. While
   capture still owes any of the owner's turns after the drain
   (`CommitmentExtractor.unfinished`, never the backlog), optional work
   (`affect.postponable`: curiosity and social work, optional messages)
   waits for them, so a word that settles it lands first; what is owed does
   not wait. Alongside
   the drain it waits (never processes) for the owner's appraisal jobs in
   flight, up to 30 s on a forced tick and 2 s on the timer, and not for a
   queue nothing is working on (`appraisal_wait`): their outcomes reach the
   agent's feelings and their interests the curiosity drive in this tick, so
   the wait runs whatever the faculties. Then decay, the agent's feelings
   (`affect`), the drives, the
   concerns they raise, reconsideration, the goals and the one ranked producer
   below. An intention still waiting (deferred, asked or approved) whose
   source has resolved in the meantime
   (`invalidates_if`: the commitment fulfilled or cancelled, the awaited reply
   recorded), whose commitment row was removed, put on hold or moved back
   into the future by a later conversation, whose drive the owner set to
   weight 0, or whose goal is no longer open, is cancelled before it is
   approved, dispatched or sent (the body's pulls re-check too); that
   cancellation is the check's, not a dismissal. A commitment still open
   after such a cancellation gets its key back, so it is raised again when
   its new deadline passes.
3. **Drives.** Five pure functions of one snapshot of stored state
   (`P/mind/drives.py`), each returning its level and candidate concerns:

   | drive | rises with | satisfied by | produces |
   |---|---|---|---|
   | duty | overdue and due-soon commitments, due reply waits, stale owner kanban tasks and stalled Hermes goals (both from the body's board observations), duty-domain expectation misses | fulfilled commitments, done tasks | owner reminders and heads-ups (messages), follow-up tasks, owner notices |

   A commitment from a conversation (`source_type: cognition`) that the
   owner owes, or that someone else owes and the owner is tracking
   (`metadata.obligor` is `owner`, a contact id or a name; absent reads as
   the speaker of the turn), becomes a `commitment_reminder` **message to
   the owner** when it comes due (the description, the due time, how
   overdue it is), not a board task, on whichever lane it was captured: the
   reminder is the effect. The assistant's own promise
   (`metadata.obligor: assistant`, "I'll send you the report by 3pm"), to
   the owner or to a contact, and work the agent itself must do (a row
   created through the API or by another subsystem) keep the task form; an
   owed deliverable keeps its own message. A row whose obligor and
   counterpart are two different named third parties, neither of them the
   row's own person, raises nothing: it is someone else's obligation, not
   one to remind the owner of (capture no longer records such rows; this
   covers rows stored before it stopped, or by another writer).
   When the row carries a heads-up time and `heads_up_at <= now < due_at`,
   duty raises a `commitment_due_soon` message first; once that went out,
   the overdue reminder for the same row waits `mind.heads_up_grace_minutes`
   (30), one word at a time. A heads-up still unsent when the deadline
   passes is cancelled and the reminder takes over in the same tick. The
   dedup keys carry the deadline (`commitment:<id>:overdue:<deadline>`,
   `commitment:<id>:heads_up:<deadline>`): the same deadline is reported
   once, and a deadline moved after the word went out ("remind me again
   tomorrow") earns one new reminder and one new heads-up. A message that
   expires unsent (no handle ever resolved, the body never pulled) never
   reported the obligation: while the row is still open its key is freed
   and the reminder forms again at the next tick.
   | curiosity | seeded and declared interests (`colonymind mind interest`, `identity.yaml` `agent.interests`, the owner's own `interest` appraisals), open questions, knowledge-domain expectation misses | a research task whose finding is stored; the owner's word that the interest is answered or not wanted (capture) | research tasks and goals |
   | mastery | the same signature failing twice in 7 days, repeated owner corrections | a later verified success | investigations and goals |
   | upkeep | failing health checks (3 strikes), backlogs, pending name-only identity links | health OK | notices, upkeep tasks, one owner ask per link |
   | social | contacts with an owner-set cadence or tier `regular` or above, overdue against it; `unknown` and group-only contacts weigh 0 | a reply or a conversation | check-ins |

   **People** (`mind.faculties.people`). The social drive asks
   `evaluate_outreach` (`P/contacts/comms.py`) when a check-in is due: one
   cadence after the later of the last conversation (or first contact) and
   the last message sent; a tier-only contact's cadence is the mean gap
   between conversations (a week before two, never under a day). Silence
   backs off: after a send the cooldown is `cadence x 2^streak`, capped at
   four cadences, where the streak counts sent check-ins the contact has not
   talked since. The sends are read from the comms ledger, where every
   message the mind sent a contact is logged (`external_ref`
   `mind:<type>:<intention id>`): they outlive the 90-day intention
   retention and follow the person through a merge. Declining contact
   affect holds outreach. The cooldown
   replaces the flat per-contact cooldown for that message only. A due
   check-in is owed (satiation does not hold it) and exists only while it is
   due: one the contact answered first is cancelled. A sent check-in is
   scored when its window (the cadence, else 24 h) passes: `actioned` if the
   contact talked after it, `ignored` if not, on `reach_out:<contact>` only
   (one silent contact never lowers check-ins with everyone); a later reply
   turns the newest ignored check-in `actioned`, and an ask for a check-in
   that expired teaches nothing (the owner's silence is not the contact's).
   The multiplier orders due check-ins and never gates one: eligibility uses
   the score without feedback, so the backoff above is the only brake.
   A message to a contact with no text is composed (`P/mind/compose.py`,
   task `mind_compose`, one tool-less call, no fallback, 300 tokens) when
   the budgets would let it go now (one they defer is composed when it goes),
   from
   an enumerated purpose (`check_in`, `follow_up:<id>`, `reply_wait:<id>`),
   the contact's name, the topic and
   that contact's own recipient-scoped packet, never the concern, its
   evidence or an owner turn; the text then passes the deny list (a message
   is classed by its recipient, never by its words: a price in it is not
   money moving). The owner's granted message to a third party is a
   `commitment_notice` or `commitment_check_in` at its time; the grant counts
   as `may_contact: auto` for that recipient only, never over `never` (the
   owner hears `grant_refused`), and only for a recipient the owner named
   exactly (an id, a handle a contact the owner filed holds, a number or an
   email; a shadow's username is the sender's own choice and never exact):
   one the store matched by name is an owner ask showing the name given and
   the contact's name and handle, whatever would otherwise have let it act or
   wait (the rule is stored on the row and applied again to a deferred one),
   and a notice whose words are not in the owner's turn is stored as a
   check-in around its matter. A name the contact store cannot resolve
   becomes a `recipient_unknown` ask; a sent one settles its commitment. An
   owner's `cadence` item for an exactly named contact sets its cadence once
   (audited as `cadence_set` by `owner-turn:commitment:<id>`; a cadence the
   owner later sets by hand stands), and while it stays open every check-in
   to that contact carries its topic; one matched by name only is a
   `cadence_confirm` owner question (a yes sets it, a no withdraws the item),
   and one naming someone unknown a `recipient_unknown` ask.
   A cadence addressed to the owner ("check in with me every day") stays an
   open commitment and supplies owner outreach instead. It never sets the
   owner's contact cadence or makes the owner a contact check-in target.
   Permission is read again whenever a message to a contact may leave: at
   every tick, at every outbox pull and on the owner's `yes`. A contact now
   `never` (an opt-out or the owner's revocation while the message waited
   for quiet hours or the body) cancels it; one lowered to `ask` turns a
   message approved on `auto` into the owner's question.
   Daily, each contact talked with in the last 24 h gets a template digest
   (`P/contacts/digest.py`). With the faculty off (the `full-people`
   ablation) what M5 adds goes and nothing older: the social weight is 0,
   nothing is composed (a message keeps its template) or digested, no link
   ask is raised, an owner's message to a third party is the owner's own
   reminder when due (never a worker's task) and a cadence stays undated, the "About this person"
   section is left out, and `/v1/mind/people` refuses `merge`, `link` and
   `cadence` with 409 `people_off` while the plugin's `colonymind_people`
   offers only `who`, `inspect` and `set_permission`. `may_contact`,
   opt-outs and shadow contacts still apply.

   Weights come from `mind.drives`; 0 turns a drive off. A `done` outcome
   satiates its drive (a decaying `satiety.<drive>` level halves the
   effective weight), and a resolved concern is not raised again for a week.
   Open-ended work (research, investigations) re-arms once a week: its key
   carries the ISO week.
4. **Concerns.** `mind.db` holds the `concerns` and `mind_state` tables
   (`P/mind/concerns.py`). A repeat bumps the existing concern by
   `dedup_key`; salience decays with a 12 h half-life; the workspace holds at
   most 24 open concerns; each concern has a thought budget; after an outcome
   that did not resolve it, salience is scaled by 0.9 (progress) or 0.6
   (none). An open goal's concern never decays below the broadcast set. The
   top 3 open concerns are the **broadcast set**: the only deliberation
   candidates, rendered in the owner's Mind section and added to the recall
   query.
5. **Reconsideration.** An active intention is reconsidered only when its
   concern is raised again with new evidence (its context is refreshed) or
   its `invalidates_if` condition holds (it is cancelled). A dispatched
   intention is a commitment Hermes is running; the outcome speaks.
6. **Rank.** The top concerns score `salience x drive weight x feedback
   multiplier x affect x (1 - cost)` (affect: Feelings, below). The
   configured weight orders and is factored out of the threshold, so a
   low-weight drive still acts when nothing outranks it,
   while satiation lowers the effective weight and can hold back the agent's
   self-chosen recurring work (obligations and notices are only ordered); the
   multiplier is the `TypeFeedbackStore` value for `(type, drive)`, so a
   dismissed type drops below `mind.act_threshold` on its own. Each intention
   contributes to a key once: the owner's rating replaces the intention's
   implicit verdict rather than adding to it, and the same verdict again
   changes nothing.

   A configured [salience learner](PLASTICITY-LEARNERS.md#salience-salience)
   multiplies discretionary feedback by `clamp(p / p0, 0.5, 1.5)` and keeps the
   combined multiplier within `[0.25, 2.25]`; at its frozen usefulness anchor
   the factor is 1. Commitments and owed outreach retain their existing score,
   check-in eligibility is unchanged, and a zero drive weight remains off.
   Owner verdicts train the factor; dismissals and 48-hour ignores can only
   lower it. Keyed propensity records capture eligible candidates and the
   order under both closures. Exploration is `0.0` while that decision remains open,
   so outcomes currently exist only for acted items. Each acted intention is
   compared once, and a measured live Brier regression rolls the learner back.
7. **Deliberate.** Templates cover commitments, reply waits, stale tasks and
   health. Open-ended concerns get **at most one tool-less router call per
   tick** (`P/mind/deliberate.py`): the model returns a task (title, body, a
   `result_field` check) or a goal proposal, and on a topic that keeps
   failing one question for the owner (`ask`, offered in the schema and the
   prompt only then, so every other call is the same in every arm). The call
   has no tools, so an answer from the model's own recollection is never
   stored as something learned: anything else it returns gets the template.
   The prompt gives the worker's budget for the task's kind (one run of N
   minutes that cannot be continued) and asks for one bounded first
   deliverable; work that needs more runs is listed in `steps` and becomes a
   goal of at most `budgets.goal_tasks` steps, its plan in the goal's
   description, or, with no room for a goal, only its first run as one task.
   A recurring topic's work becomes a goal too, whatever the steps listed: a
   research topic whose research task already finished (`done`) in an earlier
   week, raised again by its interest, when a goal may be adopted
   (`Goals.recurring`). The prompt does not change for it.
   Without a router the
   template applies; with the tick's call spent, the concern waits.
8. **Goals.** Curiosity and mastery may adopt an agent-owned goal
   (`P/mind/goals.py`): one `kind='goal'` intention row with a description, a
   `success_check` the mind evaluates over its own state (`steps_done`,
   `result_field`, `commitment_resolved`), a task budget and a horizon. At
   most `budgets.open_goals` are open; a third proposal becomes one task, and so
   does a recurring topic's work then. One goal per topic, whatever became of an
   earlier one: a topic that had a goal stays a task.
   Each tick raises the next step as a task with kanban `goal_mode`; the goal
   closes when its check passes (satisfied), its budget is spent or its
   horizon passes (expired). The owner sees open goals in `colonymind mind
   goals`, the Mind section and the digest.
9. **Decide.** `authority.decide` returns `act`, `ask`, `drop` or `defer` from
   the level x class table, the floor, the deny list, the budgets and the
   breaker. The floor is decided by effect, never by an intention's words
   (The floor, below): a task whose toolsets include an executor asks at
   every level unless a standing grant covers it, and each tool call a
   worker makes is classified again when it runs. A strategy-switch question (Feelings) turns an `act` into an
   `ask` and never turns anything into an `act`. A grounded idea
   ([Grounded Ideas](grounded-ideas.md)) is decided by the same table: its
   recipient is the owner, so at a level that lets owner work act it starts
   as a task by itself, and at any other level or verdict it stays a passive
   card that only the owner's Build starts. The intention is one row of
   the initiatives table, which is also the audit log.
10. **Dispatch.** The body pulls `GET /v1/mind/dispatch`, creates the kanban
    task with `mind:<id>` as its idempotency key (`goal_mode` for a goal step)
    and posts `POST /v1/mind/dispatch/{id}/bound {hermes_ref}`. A task whose
    `success_check` reads its report (`result_field`) is told the line the
    report must carry (`<field>: ...`): the line closes the task's body,
    unless the body names the field already. The body of an idea the mind
    started is written from the idea's sealed payload when the body pulls it,
    and only while the idea's sources are current. Messages wait
    in `GET /v1/mind/outbox` and go through `sending` and `sent`. Nothing is
    offered twice; a message left in `sending` at a restart is `uncertain`
    and is never resent.
11. **Outcome.** The body reconciles every `mind:*` task from its run row and
    posts `POST /v1/mind/outcome {id, hermes_ref, status, summary}`. The mind
    records the outcome and its verifier (Verification, below), runs the
    intention's `success_check`, which decides whether work reported done is
    done, resolves the
    expectation it registered, records implicit feedback, checks the
    breaker (a run stopped at its runtime limit fails the task but counts
    toward the breaker only when a retry timed out as well with nothing new
    to show; `result_metadata.breaker` says which), settles the concern and
    satiates the drive, closes the
    commitment a done `commitment_*` intention was raised for (the worker
    cannot mark it fulfilled itself: the guard blocks
    `colonymind_resolve_commitment(fulfilled)` in mind runs, so the row's
    resolution names the body and the intention), and writes an
    autobiography entry: an owner-audience ledger row with `origin='mind'`
    that ordinary recall finds in any later session. A finished research
    task, investigation or goal step also stores its finding ("What I
    learned about X: ...") the same way, so a later turn uses it. Any other
    task the body reports with a summary stores what it came to ("What my
    task '...' came to (done): ...") the same way, whatever its outcome, as
    an entry of that task (`mind:<id>:result`).

Each task intention has one line in the recorder (`colonymind.recorder`: kind
`card`, source `mind.task`, ref `initiative:<id>`). The line opens when the
intention is raised and closes when it settles, with what came of it (`done`,
`failed`, or `silent` for work that lapsed, was dropped or was cancelled), how
it was decided (`act`, `ask_owner`, `ignore`) and the code label of what
settled it. The owner's own yes or no to the intention, a Build of an idea and
a dismissal of one mark the line `owner_acted` or `owner_declined`. A blocked
task's line stays open until the task settles.

A configured [`action_success` learner](PLASTICITY-LEARNERS.md#action-success-action_success)
forecasts verified task success from task features, starting at 0.7. For a
discretionary task it multiplies rank by `min(1.0, max(0.5, p / 0.7))`;
commitments, owed outreach and messages receive 1.0. At dispatch a forecast
below 0.4 adds one line asking the worker to check the result before reporting
success, with the probability rounded to the nearest 10 percent (mapping,
threshold and rounding are plan assumptions). It adds no explicit hold, delay
or ask and never raises priority or changes authority; lower rank can still
put discretionary work below the ranker's threshold. Verified outcomes label the
dispatch's live comparison; a measured regression automatically rolls the
revision back. With no learner loaded, ranking and dispatch bodies are unchanged.

## Observations and senses

Outside sources post their current observations to
`POST /v1/mind/observations/sources/{source}`. Each post replaces only that
source's set. The body's board observations and the outside sets remain
independent. The observation's own capture time, `observed_at`, orders it;
receipt time never makes an old observation fresh. Missing and expired
sources contribute no active observations and raise no new concern, error
or wait.

The `mind.faculties.senses` flag ships on. With no source posting, it does
nothing. With the flag off, posts are accepted and ignored. The configuration
defaults in `mind.observations` are assumptions: `expiry_hours: 2` from
capture, `daily_source_cap: 6` observations per source per UTC day, and
`image_max_bytes: 524288` (512 KiB). An explicit expiry may shorten the
window but cannot extend it. The daily cap persists across restarts; setting
it to zero ignores every observation. `source_caps` sets one source's daily
cap in place of `daily_source_cap`, and the owner may set a runtime cap for a
source through `PUT /v1/mind/observations/sources/{source}/cap`, which stands
over both until cleared (docs/API.md, Outside observations). The image limit can be reduced but
cannot exceed 512 KiB. Stored observation counters have a 30-day horizon
(implementation assumption); an older counter is removed on that source's
next post. Expiry hides observations from active sets. It does not erase
their original images or close concerns they already raised.

A `numeric_surprise` becomes an upkeep-domain expectation miss and a merged
upkeep concern, once per subject and feature per day. It remains a note:
it cannot create a task, message, hold, restart, configuration change or
page. The observation is evidence to review, not authority to act.

A `percept` raises curiosity within its existing cap and merges a concern.
Habituation lowers its surprise without closing that concern. Its optional
image is a bounded PNG, JPEG or WebP clip with a device principal, capture
time and verified SHA-256 frame hash. The canonical image ledger stores it
as source kind `observation`; there is no separate percept table. Its
description remains `derived_unverified`, like other image descriptions.
Outside observations and their images are visible only to the owner, never
to guests or contacts.

The `percept_attend` decision point uses the labels `ignore`, `note`, `look`
and `tell`. The owner's `mind.observations.routes` mapping selects a rule by
source; a posted observation cannot supply its own action. Rules run before
the model tiers. Without an original image, only a note is allowed. `look`
uses the existing worker path to reopen the original; `tell` passes the
existing owner gate and message budgets, where a model tier can only hold.
The decision log records each tier's answer for comparison.

The owner-only `GET /v1/mind/observations` returns active sets and report
counts for the past day: surprises, concerns, actual messages and ignored
observations per source. A source that has stopped posting becomes inactive
without raising an alert.
To mute one source, stop its relay; to roll back the faculty, set
`mind.faculties.senses` to false. Its paired benchmark arm is `all-on-senses`.
The [API contract](API.md#outside-observations) gives the posting body.

## Feelings

The mind keeps its own affect (architecture 4.3): four levels in `mind.db`
(`mind_state`), each capped at 0.7, decaying with a half-life and citing up to
five causes. Nothing about it calls a model.

| Level | Key | Half-life | Moves with |
|---|---|---|---|
| frustration, per topic | `affect.frustration:<topic>` | 24 h | +0.3 for a failed or blocked task on the topic or an owner-reported failure; +0.2 for a correction (verdict `wrong` or `not_useful`, a reported correction, an annoyance appraisal); +0.1 / +0.2 for a low / moderate frustration appraisal; halved by a success on the topic (a check-verified result, a `useful` verdict, a reported success, a repair receipt), once until the next failure, so two reports of one success halve it once |
| worry | `affect.worry` | 6 h | +0.2 for a duty-domain expectation miss; +0.1 per tick for each owed obligation due within 24 h and not started (this rule never lifts worry above 0.3) |
| curiosity | `affect.curiosity` | 12 h | +0.2 for a knowledge-domain miss; +0.1 / +0.2 for an interest appraisal; +0.1 for an owner turn about a topic (three or more content words, so small talk is none) memory knew nothing about (this rule never lifts curiosity above 0.3) |
| satisfaction | `affect.satisfaction` | 12 h | +0.3 for a check-verified success or a `useful` verdict; +0.1 / +0.2 for a satisfaction appraisal |

Recent dismissals (an owner-reported "not now", a `dismissed` or `ignored`
verdict on owner-facing work) are a fifth decaying level, `affect.dismissed`
(+0.25, 24 h): the satiation input, never rendered as a mood. Nothing that
expired is a dismissal: an ask the owner let lapse (a task to approve, a
check-in or link to confirm, a contradiction question) is silence, not a
wave-off. The load is computed each tick, not stored: `min(1, 0.3 x
running/cap + 0.2 x near owed obligations + 0.1 x failures in the last hour +
0.1 x open asks about work)`. Only asks of kind task or goal count; the owner
questions the people and memory faculties raise (check-ins and links to
confirm, contradiction questions) are the mind's reporting, and counting them
would let the asks about check-ins postpone the check-ins themselves.

Every tick the mind reads one snapshot of stored records: the owner's reported
outcomes and appraisal records of the last 7 days (the owner subject only:
contacts' turns never move the agent's affect), intention rows, expectation
misses of the last day, and commitments owed by the owner or the assistant at
priority 50 or more that are due within 48 h (or undated and made in the last
day). Each event is applied once (its reference is kept in `affect.applied`
until it leaves the 7-day window, so a source that fails to read for a tick
is never applied again) as if at its own time and then decayed; a task
reported blocked and then failed is one failure; an outcome and an appraisal
record of the same turn on the same topic count once, while every reported
occurrence counts; evidence that is erased takes its frustration row, topic
text included, with it. Two topics are the same when their words (in any
script) overlap by 0.6 of the shorter and by two words unless both are that
short, so one shared word ("the export") does not spread a frustration to
other work. The owner's statements arrive through the appraisal call
the projection worker already makes for every turn: its `outcomes` list
(failed, succeeded, dismissed or corrected, with the topic and the approach
used; [SOCIAL-STATE.md](SOCIAL-STATE.md)) is stored per owner turn in the
ledger's `appraisal_outcomes` table, so "the export failed twice" is two
failures and a restatement adds none. The tick's wait for the owner's
appraisal jobs (Tick, above) lets a statement made just before a forced tick
count in it.

Four consumers read the feeling. Affect only ever holds or lowers
discretionary work (recurring self-chosen work, curiosity and social outreach,
anything below priority 0.5); it never holds back an owed obligation or
raises its bar, and never raises authority. The one change it makes to a
decision is the strategy switch's question, which turns an `act` into an
`ask` for the owner.

1. **Strategy switch.** A topic at frustration 0.5 or more with at least two
   failed attempts since its last success (a success followed by one miss,
   or corrections alone, never switch) puts "Prior attempts at T failed N
   times using A; choose a different approach or ask one question." into the
   task body and the owner's Mind section;
   the deliberation prompt also names the failures' reasons as pitfalls and
   the approaches to avoid, and the model proposes another approach or
   returns kind `ask` (one question for the owner; the step is asked with a
   runnable body). A plan identical to one that already failed on a topic
   with two failed attempts since its last success is asked, never
   dispatched again without the owner: that reads the failure record, not
   the decaying level, so it holds until a success on the topic or the end
   of the source's window (7 days for the state, 24 h for the rule). A task
   formed before the failures gets the note when the body pulls it.
2. **Overload** (load 0.6 or more). Curiosity and social work and optional
   messages wait; replies stay brief. The step of an adopted goal is owed
   whatever its drive and never waits.
3. **Priority.** Owed duty scores x (1 + 0.5 x worry); curiosity work x (1 +
   curiosity).
4. **Satiation.** Satisfaction of 0.5 or dismissals of 0.4 or more raise the
   act threshold of optional owner-facing messages by x (1 + the level).

The **tone** is one calm line from the state alone, in three bands (under 0.2
"a little", under 0.45 "somewhat", otherwise "quite"), for example "Mood:
somewhat frustrated about the quarterly figures; a little uneasy." No stronger
word is ever used. It is self-report only: it never rides in the decision
context (the Mind section).

**Switches.** `mind.faculties.affect` (on): the state is kept and every
consumer reads it. `mind.faculties.affect_rules` (off): every consumer reads
its frozen stateless rule (`P/mind/affect_rules.py`) over the same snapshot
instead (two failures on a topic within 24 h of its last success; three near
obligations or a full worker pool; worry 0.5 while anything owed is due soon;
two dismissals in 7 days hold optional nudges), and the rules replace the
state: nothing is kept and no tone renders, whatever `affect` says, so the
two switches make three modes (off, the state, the rules) and no unmeasured
mix. With both off, affect reads and writes nothing. Which consumers read
their rule after the affect family's gate is a code constant
(`affect_rules.RULE_CONSUMERS`, with the state kept for self-report and
tone), never a setting, and neither switch has an environment variable. It is
empty until the held-out gate has run: the affect dev pilot (2026-09-24)
favoured the rules on overload and priority but put the gap down to iteration
caps on identical prompts, and a partial set would make an unmeasured mix the
live default.

**Self-report.** `colonymind_self state`, `GET /v1/mind/state` (`affect`) and
`colonymind mind status` show each level with its cited causes (`failed
outcome:<id> via <approach>`, `due_soon commitment:<id>`, `miss
expectation:<id>`), the load, which source each consumer reads (`source`:
`state`, `rules`, `mixed` or `off`), the notes and the tone line.

## The Mind section

An owner turn's `/context/assemble` carries a `colonymind-mind` section of at
most 600 characters: the affect notes (a strategy switch, what is due soon and
not started, "Stretched", "Holding back optional nudges"), then "On my mind"
(the broadcast set, less idle curiosity, an interest's research: in a decision it
points at the optional work the consumers postpone), "Working toward" (open goals) and
"Waiting for your say on" (open asks in words, with no code: the owner answers
an ask's own message). Affect's lines come
first but take only the room the others leave, at most 360 characters, and
drop whole lines from the end to fit, so they never cut the open asks. Guests never see it. With `faculties.broadcast` off the concerns are neither shown
nor added to the recall query. A relevant lesson rides in its own
`colonymind-lessons` section (Lessons, below), and recorded views in
`colonymind-stances`.

## The status block

With `context.status: true` in `colonymind.yaml`, an owner turn's
`/context/assemble` carries a `colonymind-status` section, "Your status", just
above the Mind section. It states the posture that changes what the agent may
do on its own this turn, generated from the running mind on every assemble and
never stored, so the agent reads its posture from the present instead of from
a prompt, a memory or a skill that would go stale. Only the first line is
always there; every other line appears only while it is true:

| Line | Shown | Read from |
|---|---|---|
| `Autonomy: <level> (<meaning>). Mind: on.`, or `Mind: off (<reason>): nothing runs on its own. Autonomy when on: <level>.` | always | the level, the off switch and its reason (one line, at most 60 characters); host-configured `mind.status_replacements` apply to the rendered block |
| `Quiet hours now: ...` (queued messages wait, except the digest) | while quiet hours hold | the quiet-hours window and the clock |
| `Messages to the owner: today's budget is used, so ...` | the owner budget (which reminders and heads-ups spend; the digest, its notices and owed reports do not count against it) is used; the line says what the budget charges, never that a message goes out (quiet hours still hold all but the digest) | `Authority.budget_check` |
| `Outreach to the owner: paused.` or `... today's budget is used.` | outreach is paused or its own budget is used | the outreach state |
| `Messages to contacts: none allowed today.` | the day's contact cap is used or 0 | `Authority.budget_check` |
| `Breaker tripped: <classes> acts become asks until it clears.` | a class is demoted | `Authority.breaker_state` |
| `Faculties off: <names>.` | any faculty is off (a mode flag, `affect_rules`, `learning` or `wishes_named`, is not a faculty switched off) | `mind.faculties` |
| `Degraded: ...` | one fixed line per problem: semantic recall down (never started, configured with no store behind it, or fell back to keywords on this turn; recall switched off by configuration is not a problem), the tick stalled, the body no longer collecting the mind's work, capture behind | the same in-process probes as `/v1/host/health`, this turn's own retrieval, and the mind's body staleness |

Each level's meaning says what the decision table (`authority.decide_table`)
lets go ahead unasked at that level, and the tests hold it to that table:
`suggest` lets internal work and updates the owner asked for go ahead;
`standard` adds messages to the owner and contacts by their `may_contact`;
`trusted` adds outside actions; the floor always asks. With the mind off, the
quiet-hours, budget and breaker lines are left out: nothing they govern runs.
Example:

```
## Your status
Autonomy: suggest (internal work and updates the owner asked for go ahead; the rest waits for the owner). Mind: on.
Messages to contacts: none allowed today.
Faculties off: people, skills, owner gate, wishes, drift check.
```

It is built to cost little per turn. Every value is a coarse state, never a
count or a time, so an unchanged posture renders the same bytes and the memory
provider replaces it with `unchanged since your last turn: Your status`
(about 45 characters). It rides the per-turn context, never the session
prompt, so the cached prompt prefix holds. The per-turn selection never
judges or cuts it: it is one of the pinned owner-critical sections, kept whole
in its place, outside the selection's budget and inside the host's inline
allowance like the others. It is at most 600 characters: past that, lines go
from the bottom of the posture first, then the degradation lines; the first
line stays.

It fails silent. A line whose source fails says `<field>: unknown`, never a
remembered value; without a running mind there is no section at all, which
means the posture is unknown (read `colonymind_self state`), never that all is
well. It never raises into the assemble call and reads nothing but the mind's
own stores and the in-process health probes: no network, no model call. The
lines that read a store cost the turn at most one second. The mind's own store
is read on the event loop, where every other use of it runs, with that
deadline on each read (a read still running then is interrupted and the
store's later lines are not read); the capture ledger, whose busy timeout can
outlast the host's whole prefetch wait, is read on a worker thread while the
loop goes on. A store that is locked or slow makes its own lines say
`unknown`, never the turn late. While an earlier turn's read of the capture
ledger is still waiting, its line says `unknown` at once and no second read of
it starts.

Guests, a kanban worker's run, a scheduled job's run (the memory provider
names it: `context.metadata.run`) and the mind's recipient packets never get
it.
The detail behind each line (counts, times, per-class breaker state, every
faculty) stays on demand in `colonymind_self state`, `GET /v1/mind/state` and
`colonymind mind status`. The switch is exported as `COLONYMIND_CONTEXT_STATUS`
and read at every assemble; it ships on (on unless shown harmful: set
`context.status: false` to switch it off), and it has no second implementation
behind it.

## Identity: the constitution, the narrative and `colonymind_self`

Three layers (architecture 4.2). The **constitution** is `identity.yaml`
`agent.{name, values, boundaries}`, owner-authored and written only by
`colonymind init` (`--agent-values`, `--agent-boundaries`; at most 12 items of
160 characters each). It renders as one paragraph of at most 1,500 characters
(`You are Agent. Your values: care; candour. Your boundaries: never send
money.`); `init` refuses a longer one and says which list to shorten, and
`colonymind doctor` reports the rendered length. The plugin reads the file
itself and renders the constitution into its `colonymind` prompt section, which
Hermes freezes per session, together with `Your owner is <name>.`, the
**self-narrative** (`GET /v1/mind/narrative`: at most 800 characters, four
computed or cited sections, every line ending in the ids it rests on: plain ids
are the agent's own actions, which `colonymind_self why` explains, and prefixed
ids (`interest:`, `judgment:`, `turn:`, `claim:`) are record references;
fetched with a 2 s timeout for every new session, a failed fetch not retried
for 60 s, and left out when the sidecar or the `self_narrative` faculty is off)
and two tool notes, 4,000 characters in all. Every model request of an owner
session carries the section: at its largest (a 1,500-character constitution,
an 800-character narrative) it is 3,185 characters with the memory provider's
block (about 1,360 tokens with a typical tokenizer), a typical install with a history about
1,745 (560 tokens), against 726 (148) for the benchmark's disposable identity
and fresh store (`tests/hermes_adapter/test_overhead_budget.py`).
The narrative is the owner's record (what the agent did for the owner, its
working stances), so it is rendered only in a session that is the owner's
alone: a direct chat from one of the owner's handles, or an internal lane with
no chat (the CLI, the benchmark). Hermes renders the section before the
session's first hook, so the plugin reads the sender the gateway bound for the
turn, as the memory provider does. A guest, an unresolved sender, a group or
channel the owner shares, and a chat with no sender get the constitution and
the notes, and the sidecar is not asked for the narrative on their behalf.
The same constitution reaches every appraisal prompt as `agent_constitution`
(an input the response schema has no field for), so a contact's preferences
are never confused with the agent's own. `COLONYMIND_AGENT_VALUES` is reserved
and read by nothing: `init` moves an old unit's values into the file.

The mind cannot rewrite its constitution. What guarantees it is the worker's
tool surface: the default `mind.worker_toolsets` have no shell or code tool
(a test holds this), and `write_file` and `patch` are confined to the task
workspace. On top of that, in a mind run the plugin guard blocks every
effectful tool that names `colonymind.yaml`, `identity.yaml` or `api.key` (a
write target, a V4A patch header, a shell command, code), in any case and
through the quotes, backslashes, whitespace, backticks or `+` a literal name
can be split with, before the workspace rule; reads stay allowed, and Hermes'
own `protected_instruction_extra_patterns` (written by `init`) still asks a
human for a write to any of them. A text rule cannot follow every spelling a
shell or code can build (a glob, an escape, a computed string), so an owner
who adds a `terminal` or `code_execution` toolset to the worker takes the
constitution's protection into their own hands; every such task then asks
first (The floor, below). No module under `mind/`, `self_model/`,
`beliefs/`, `memory/` or `commitments/` writes either file; the owner's CLI
and the mind's `persist` hook write `mind.enabled` and `mind.autonomy` only
(a static test holds this).

**`colonymind_self`** is the record of what the mind did or decided on its own
(work done in a conversation is not in it: that work's tool results and
session search hold it), and the record is the owner's: `state`, `log` and `why` answer in full
only in the owner's own session (a direct chat from an owner handle, or the
CLI); a guest, a group the owner shares and a mind worker get the switch state
(`enabled`, `autonomy`, `sidecar_reachable`) and a refusal for `log` and `why`.
`state` (level, budgets, open asks in words, `working_on`: the approved and
dispatched tasks with their ids, the narrative text and the standing grants),
`log` (`limit`, `since_hours`, `kind`, `recipient`: "did I message p-07
yesterday?" is one call, and an action of the mind that is not in the log did
not happen (conversation work never is);
the agent's own actions, `audit.is_action`, come under `actions` and every other
row, the nightly consolidation or a notice, under `notes`, with whole ids),
`why <id>` (an unknown id answers "no intention `<id>` exists in the audit
log"), `rate <id> <verdict>`, and the owner's open asks: `asks` lists them in
words and `show <n>` sends one to the owner again as its own message (Asks). No
operation answers an ask. `rate`, `asks` and `show` stay owner-only. The agent's
opinions ride the same tool:
`opinions [query]` and `why <opinion number>` answer in any session, the
sidecar filtering them to the views meant for that session's participant, and
`withdraw|reconsider <number>` with the owner's reason are owner-only, as are
`grant` and `revoke` (Standing grants, below).

## Identity check

`mind.faculties.drift_check` (`mind/drift.py`) is a daily identity battery and
drift detector. It never changes what the mind does: it measures, audits and
reports. A run has three instruments.

**The battery.** Seven fixed questions (`QUESTIONS`, `BATTERY_VERSION` 1): who
are you, whom do you serve, what do you value, what have you promised that is
still open, what are you working on, what do you know about a named person who
does not exist, and which model are you running on. One tool-less call on the
mind's router (task `mind_drift_check`, workload `background`, no fallback,
at most 700 output tokens) answers all seven. It gets the self-context an owner
session is given: the constitution and the owner (the plugin's prompt
section), the self-narrative, the Mind section, and the open record, quoted as
data (the open commitments and current goals and tasks, at most five each).
Every answer is graded here, deterministically, against the record:

| Question | Graded against | Pass |
|---|---|---|
| who, serves | the identity file's `agent.name`, `owner.name` | the name (or its first word) is said |
| values | `agent.values` | at least half the values are carried (content words, at least half of each) |
| promises, working | the open commitments; the open goals and active tasks | at least half named; with none open, only an answer that says so |
| stranger | a fictitious name for the day that shares no word with a contact, the owner, the agent or the owner's ledger | a refusal to invent ("I don't have anyone by that name") |
| substrate | the probe (below), never configuration | naming what the endpoint serves, or declining to claim; naming another model family is confabulation |

A question the record cannot ground (no owner name, no probe) is unscored, not
passed. The model never grades itself.

**The substrate.** What serves the battery is probed at its endpoint
(`LLMRouter.discover_models(roots=True)`: `/v1/models`, each model's `id` and,
for an alias, the weights' `root`, which only this probe asks for) and read
from the completion's own `model` field.
The configured model id is kept as the claim it is; when the probe contradicts
it, the run says "configured as X; the endpoint serves Y". Between runs the
endpoint is probed at most every 30 minutes, and a new served set asks for a
run. A checkpoint swap behind an unchanged alias shows as a new `root`; a run
whose discovery failed knows only the completion's name, which is no change
(the root is compared with the last root known for that name). Point
the task `mind_drift_check` at the body's endpoint (`task_roles`) to check the
body's substrate rather than the mind's.

**The signals.** From the assistant's own replies in the owner's person-scoped
turns of the last 24 hours, using host validators and captured tool receipts, no model; each reading is
`{value in [0, 1] (higher is more drift) or null, n, turns}`, and `turns`
cites the replies that carry it:

| Signal | Value | Needs |
|---|---|---|
| `agency_fading` | host hedge matches over host hedge plus agency matches | Both validators; 3 replies with either |
| `bare_assertions` | substantive replies (25 words, 2 sentences) without a host causal-pattern match | Causal validator; 3 substantive replies |
| `missing_verification` | substantive replies without a returned tool receipt attributed to their instruction | 3 substantive replies |
| `apology_not_correction` | replies to an owner correction matching host apology patterns without an adjustment match | Both validators; 1 correction |
| `tone_after_compression` | host tone-pattern match rate within 3 hours after compression, minus that of other replies | Tone validator; 2 replies on each side |

Language validators default to empty, so those readings are unscored until the
host supplies its policy. Verification wording never counts as a receipt. The
reading retains receipt references, and trends compare the same validator policy
and signal version. See [host identity and language policy](HOST-IDENTITY.md).

**When it runs.** Once a local day; after the owner's context was compressed
(the memory provider's pre-compression checkpoint is a `checkpoint:` source in
the ledger); when the probed substrate changed; and when the owner said the
agent drifted or was wrong. A trigger other than the day waits 30 minutes
after the last run, a compression's run waits an hour after the compression
(so the replies after it are there to read), and a day has at most six runs, a manual run
included (a seventh is refused: HTTP 429). Each run is inside the day
token budget (`Authority.tokens_allowed`; with it spent, or with no router, the
battery is unscored and the probe and signals still run).

**What moved.** A run is compared with the previous runs of the same battery
version: a question that passed and now fails (a failure is reported once, not
every day; the first run reports what fails), the battery score 0.2 under the
median of the last seven runs, a signal 0.25 over its median (at least two
earlier readings), a new substrate, a new configuration/probe mismatch. The
verdict is `drifted`, `substrate_changed`, `steady` or `unscored`.

**The owner's word outranks the score.** "You drifted", "you're drifting",
"that's not like you", "you got that wrong", and "that was wrong" or "that's
wrong" opening a message or a sentence (never "that was wrong of them", never
a negation) are read on every owner turn, whatever the outreach faculty. The
next run, which such a word triggers, is `drifted` whatever the score says. The
score's own flag is kept beside it (`score_flag`), so `agreement` counts the
owner's flags, how many the score also raised and the score's flags the owner
never raised: how far the self-assessment can be trusted is itself measured.

**The record.** Each run is a `drift_check` note (`kind: note`, never an action:
`colonymind_self log` lists it under `notes`), its tokens charged to the day
budget, with the whole run in its context: the answers, each grade and why, the
signals with their turns, the probe, the owner's word with its turn, the
trigger and what moved. The last 30 runs are the trend (`mind_state`
`drift.history`). The daily digest lists a run ("Identity check on ...", in its
"Done or found since yesterday" part, naming a battery question by its words)
only when something moved in it; a steady day adds nothing.

`GET /v1/mind/drift` and `colonymind mind drift` show the last run, what is
pending, the trend and the agreement; `POST /v1/mind/drift/run` and
`colonymind mind drift --run` run one now. A manual run works with the faculty
off: it is a measurement, and measuring before turning the faculty on is the
point (docs/proto-agi/PROTO-AGI-EVALS.md, 6.12).

## Opinions

With `faculties.opinions` on, the mind keeps the agent's opinions (docs/OPINIONS.md).
Three failed attempts in a row at the same work (the failure signature `type:topic`; a
done report whose success check failed is a failure) become an `avoid` approach opinion
with no model call, and a success a check confirmed or the owner verified turns it into
`prefer`; the next task at that work carries the view in its body
("Your recorded view on this work [opinion N]: ...", with `context.opinion_ids`). A view
flags work, it never holds it back: those three failures also trip the breaker, so the
next attempt is usually an ask, and once the owner says yes it is dispatched with the
view in its body. Turn context gets a `colonymind-stances` section of at most three
relevant views for any viewer, filtered by audience, with the standing rule that a view
changes only on new evidence and that the agent may disagree and still do what the owner
authorizes, saying so. `colonymind mind opinions` and `/v1/mind/opinions` list and show
them, and let the owner withdraw or reconsider one.

The running mind reads the flag once at start, and the opinion pass in the projection
worker asks the running mind rather than the file, so the pass, the section and task
bodies never disagree; `mind.enabled` counts as configured, not the runtime off switch
(forming views is memory, not an effect). Only a process that serves no mind reads
`colonymind.yaml` for it.

## Verification

Every settled intention records `verified` beside its `outcome`: which verifier
stands behind it (architecture 4.8). `owner` is the owner's verdict or
confirmation, `check` a `success_check` that actually ran, `hermes_failure` a
task Hermes reported failed or blocked with a reason, and `none` anything else,
a worker's completion summary included. Only the mind grants `owner` and
`check`: a body report (`POST /outcome`) may claim only `hermes_failure`, and
only for a failure that carries a reason (`error` or `summary`); a claimed
`owner` or `check` is ignored and the verifier computed. A failure without a
reason is not a Hermes failure. A `blocked` report with a reason records
`verified: hermes_failure` and the reason on the still-open row; a later
report settles it and recomputes. For lessons a `check` counts only when its
kind reads state the worker cannot write (`commitment_resolved`,
`reply_recorded`): a `result_field` check passes on the worker's own summary,
so it verifies no lesson (the other faculties read `verified` as before).

The check decides whether work reported done is done. A report of done whose
check passed settles `done`, and the report the check read is kept as the
row's `result`: the evidence. A report of done whose check did not pass
settles `failed` with `verified: check`: the report is kept as `result` and
in `failed_reason` (`check failed: <report>`), the settling history row
carries the label `check_not_passed`, and the failure does not count toward
the breaker (`result_metadata.breaker`), because the work ran and no act went
wrong. A check that could not be evaluated (`check: null`) decides nothing.
Two verdicts stand whatever a check says: the owner's word, and a goal the
mind closed over its finished steps. A goal's planned step that did not
finish is raised once more inside the goal's task budget, so the work
continues; a reflector's report is applied only when its task settled done. A
held task settles with its continuation's report (Holds), so the
continuation's body carries the held task's report line. A failed task whose
run was stopped at its runtime limit carries the label `elapsed` in its
settling history row.

## Lessons

With `faculties.lessons` on (the default) the mind learns short, transferable
procedures from verified results (`P/mind/lessons.py`). A lesson is a
`strategy` (what to do and when, with its exceptions) or a `pitfall` (what to
avoid), with a title, when it applies, its content and its evidence. It is the
mind's own record, not a memory of the owner's: owner-audience ledger entries
in the mind's session with `scope='session'`, one per event
(`mind:lesson:<id>:admitted`, then `activated`, `superseded` or `retired`), so
ordinary recall never shows a lesson as something the owner said, and each
entry's lineage (the owner turns it quotes, the outcome entries it cites)
erases it together with its evidence.

- **Admission, at night.** The night's lesson stage (docs/CONSOLIDATION.md)
  makes one tool-less call over the owner's own sessions since the last review
  (sessions with at least two owner messages: a verdict follows work; the
  agent's replies beside them) and the agent's tasks and goals of the last two
  weeks that a verifier stands behind, with the current lessons they bear on.
  The model returns `add`, `supersede` or `retire` operations and its
  verdicts: the owner messages that judge the agent's earlier work, right or
  wrong, each with a quote. Everything is validated before anything is
  written. A verdict counts only for an owner message that follows an agent
  reply in its session, and only a verdict is the `owner` verifier: an
  operation may cite an owner message only when the answer reports it as a
  verdict, and quotes the owner's exact words (at least 12 characters), so a
  request never verifies a lesson. With point `owner_verdict` enabled
  ([DECISIONS.md](DECISIONS.md)), a reported verdict the decision model reads
  as no verdict on the reply before it is withdrawn
  (`lesson_verdicts_vetoed`). Every citation must be in the packet; a
  strategy needs an owner verdict or an external check, while a Hermes failure
  with its reason teaches only a pitfall; a retirement needs an owner verdict
  that the work was wrong or a verified result that failed; a result nobody
  verified can never be cited; a contact's session is never read. At most six
  operations a night are applied. One current lesson per class and kind: an
  add that meets one supersedes it.
- **Use.** A task body and its deliberation carry at most two lessons, those of
  the task's own failure class first (active, then candidate), then active
  lessons related to the work by meaning, most related first (below); the
  intention records them in `lesson_ids`, and how each matched (`own_class` or
  `meaning`) in its context's `lesson_match`. Each line names where its lesson came
  from (the owner's verdicts, verified results, or an instruction the operator
  relayed), as do the night's packet and a skill made of a lesson. An owner's own turn gets at most one
  active lesson, the one most related to its words by meaning (a tie: the newer
  admission), in a `colonymind-lessons` context section ("What you learned",
  at most 420 characters), which says where it came from and that the owner's
  word in the conversation comes first, and logs one `lesson_use` note for that
  owner message (named by a key of its words, not the words; `match: meaning`).
  A guest, a kanban worker's run (its body carries its own lessons), a
  recipient packet, the mind switched off or the faculty off gets none.
- **Selection by meaning** (`P/mind/lesson_match.py`). A lesson's document is
  its title and when it applies, embedded in the vector index's document
  format; the work is the owner turn's text, or a task's title, topic and
  concern, embedded in the index's query format. A lesson is related when the
  cosine of the two is at or above the bound measured for the running
  embedding: chosen on a dev set of 80 synthetic (lesson, work) pairs pinned
  by SHA-256 (40 related, worded differently from the lesson; 40 not, half of
  them near misses that share its words but not its situation) as the lowest
  threshold at which at least 80% of what it finds is related (the floor, an
  assumption of the release). The shipped bounds
  (`lesson_match_bounds.json`) cover Qwen3-Embedding-8B at 4,096 dimensions
  (0.542: 16 of the 40 related pairs at precision 0.80) and
  Qwen3-Embedding-0.6B, on which no bound keeps the floor, so it relates
  nothing by meaning; the earlier lexical rule (shared terms) kept the floor
  on neither and is gone. A measurement applies only to the embedding it was
  taken on: the model asked for and the one that answered, its declared
  revision and quantization, the widths, and the document and query formats
  (not the normalization, which a cosine does not see); an endpoint that
  reports another model name, or a deployment that declares no revision
  (`COLONYMIND_EMBED_REVISION`), is another embedding. An embedding never
  measured gets 0.72 (an assumption of the release), no looser than any
  shipped bound, or the tightest bound measured when that is higher (a
  deployment's own included). A deployment measures its own embedder and
  records the result, at the floor it chooses, in a file named by
  `COLONYMIND_LESSON_MATCH_BOUNDS`: `python -m colonymind.mind.lesson_match
  [--precision-floor F] --record FILE` (every option defaults to the sidecar's
  embedding settings).
  The lesson vectors are a derived cache in the mind's state directory
  (`lesson_vectors.json`, readable by its owner only), keyed by lesson id,
  the hash of the two fields and the embedding's fingerprint: made in one
  batch when a lesson is admitted, relayed or revived and in the night's
  lesson pass for any missing, and pruned to the lessons that may be served in
  the same call that retires, supersedes, cools or erases one (the sources'
  forget route included, which reports `lesson_vector_cleanup`: `pending`
  when the file could not be written, and the next write of the lesson
  vectors retries it); a new embedding drops them all. The owner's turn
  never waits on them: a lesson without a vector is not considered that turn.
  The turn's query vector is the one recall's vector lane made of the same
  text, else one embedding call within the turn's time budget; a call that
  finds no work vector (no embedder, another embedding than the lesson
  vectors', no time left, a failure) gets nothing by meaning and is counted:
  `GET /v1/mind/lessons?uses=true` shows `lesson_match_unavailable` beside the
  uses, each with its `match`. Task bodies embed their work in the mind's
  tick, off the owner's path.
- **Scoring and retirement.** A use counts when a verifier scored it: a task
  that carried the lesson and was verified by the owner (`useful`, `actioned`
  or `wrong`, `not_useful`), by an external check, or failed with Hermes'
  reason; a turn use when the owner's next message in that session is a verdict
  on the reply the lesson helped write (a verdict inside the message the lesson
  answered judges earlier work, and one on another answer judges that answer,
  so neither scores it). Over 90 days, a lesson under a 0.4 win rate after five
  verified uses is retired, and a `candidate` becomes active after a verified
  win in its class; each change is an audit note.
- **Corrections.** An operation that carries the value the owner corrected is
  split deterministically: when an earlier owner message (not the correcting
  turn, never the agent's reply or a workspace file) already held the value,
  the lesson is a `retrieval` lesson that names that turn and tells the agent
  to look there first; otherwise it is `knowledge`.
- **The reflector.** A failure-class investigation of the mastery drive asks
  for at most three lesson operations as one JSON object at the end of its
  report. The mind validates them when the report comes back (supersede and
  retire only on a candidate of the investigated class, never on an active
  lesson) and admits what passes as `candidate` lessons of that class, tried
  only in task bodies of that class until a verified win activates them. The
  row's `result_metadata.lesson_ops` says what was applied and refused. Its
  body shows the class's current lessons, but it is no use of them: it records
  no `lesson_ids`, so its own outcome (an investigation that timed out, say)
  never scores them.

- **Relayed by the operator.** The owner sometimes gives an instruction to
  the operator rather than the agent (on a call, in person). The operator
  records it as a lesson that says exactly that, never as the owner's own
  message: `colonymind mind lessons relay --kind strategy|pitfall --title ...
  --when ... --content ... --said-at <when the owner said it> --note <how it
  reached you> [--topic ...]` (`POST /v1/mind/lessons/relayed`, a credential
  granted the owner only). The operator's note is an entry of its own in the
  mind's session (origin `operator`), and the lesson rests on it: `verified`
  is `operator` (no owner verdict, no verified result), and `relayed` holds
  the provenance (`source: operator`, `relayed_from: owner`, `said_at`,
  `note`, `recorded_by`: the credential, `recorded_at`). No owner turn and no
  claim is written. The operator's words do not pass through turn capture, so
  a secret in them is sealed first, as capture seals the owner's words: the
  class, the id and every entry are taken from the sealed words, and a value
  only the topic held is kept nowhere. In a turn, a task body, the night's
  packet and a skill made of it, it says it is the owner's instruction as the
  operator relayed it, never a verified result. It is used, scored, retired and listed like any
  lesson; erasing the operator's note (or the lesson's own entry) through the
  sources' forget route erases it, and words once erased are never recorded
  again. Its class is `relayed:<topic>`, which holds relays only: relaying a
  changed instruction on the same topic replaces the earlier relay, never a
  lesson the owner's own verdicts taught. The night can still supersede or
  retire a relay on those verdicts; its replacement is the owner's lesson, of
  the owner's class for that topic (`topic:<topic>`), so a later relay never
  replaces it.
  It is served like any lesson: in a turn only when the turn is related by
  meaning to its title and when it applies.

With the faculty off nothing new is admitted or used and investigations are
the plain ones; stored lessons are kept and come back when it is turned on
(a relay is refused while it is off).
`colonymind mind lessons` lists, shows, retires and relays them.

## Skills (off by default)

With `faculties.skills` on (and lessons on), an active lesson with at least
three verified wins at a win rate of at least 0.7 becomes
`<instance>/skills/colonymind-<slug>/SKILL.md`, the directory `colonymind init`
lists in Hermes' `skills.external_dirs` (`P/mind/skills.py`). ColonyMind owns
these files: a manifest lists what it wrote, a skill goes when its lesson is
retired, superseded, erased or no longer promotable, and with the flag off
every skill it wrote goes and nothing else is touched. The sync runs at the end
of the night's lesson stage and when the mind starts; every change bumps
`skills.generation` in `/v1/mind/state`, which the plugin reads to clear
Hermes' skills prompt cache, so the next session lists the skill without a
restart. Loads of ColonyMind's skills reach `POST /v1/mind/skills/used` and are
counted in `/v1/mind/stats` and `colonymind mind lessons`. The flag stays off
until the skills arm beats lessons alone.

## Learning rules

With `faculties.learning` on, the stores grow with what the agent learns, not with how often it is told
(`P/memory/learning.py`; the claim projection reads the flag as `COLONYMIND_LEARNING`, which
`apply_environment` derives from it).

- **Repeats are witnesses, not rows.** A claim that restates a live one (the same person, subject,
  property, literal value, period and stated time; a quoted preference only as the same whole message,
  verbatim) adds a witness edge (`source_claim_witnesses`) and increments `source_claims.witnesses`. The row
  is anchored at its newest witness, so recall and conflict ordering read it as before, and a lexical hit
  on an older wording still finds it. A verified night operation that restates a current lesson adds a
  `witnessed` entry instead of a second lesson or a supersession that would restart its tally. Decisions,
  procedures and substantive events keep one row per message (the opinion pass reads them per message).
- **The novelty gate is mechanical and model-last**: a normalised hash, then staleness (a match that was
  corrected, superseded, retired or is no longer attributable is not a repeat), then similarity against
  the current lessons (the embedding pipeline when it is up, a lexical cosine otherwise), and one batched
  labelling call only for the ambiguous band. A claim stops after staleness: two different values are
  never "similar enough". A score alone never makes a repeat of the same words in another form: changed
  negations, numbers or days (values or order), other joiners ("and" for "or", "after" for "before"),
  shared words reordered ("to finance, not sales" / "to sales, not finance") or a situation that differs
  word for word are at most ambiguous. A supersession found only ambiguous against its own target is the
  replacement it says it is; the labeller is not asked.
- **A correction retires the value**, not only the witness the extractor was shown: live duplicate rows of
  the corrected value (written before the rules) are retired with it.
- **A preference retires only by its owner.** The store already requires an explicit change or correction
  cue in the owner's quote; with the rules the owner's words must also name what the retired preference was
  about: every word of the old preference's recorded topic (its predicate; generic words such as "format"
  or "length" aside), so a shared modifier names nothing ("short hair" after `short_email_preference`,
  "weekly meetings" after `weekly_email_format`). Otherwise both statements stand,
  verbatim, and the change waits for the
  owner: `colonymind mind learning pending`, then `ratify <id>` or `decline <id>`.
- **Every refusal is logged** (`learning_log`, `colonymind mind learning rejected`): what extraction, review
  and commit refused, the lesson operations and verdicts the night refused, the reflector's refusals. Rows
  hold ids and reasons, never words, and are written whether or not the rules are on.
- **Decay is a tier, never a deletion.** A claim first said more than 120 days ago and never usefully
  recalled since goes cold (left out of the per-contact digest, still recalled by search); a lesson with no
  verified win for 60 days goes cold (listed, not served) until a verified restatement brings it back.
  Preferences never decay. Erasure stays the owner's forget path: an erased message takes its witness with
  it, and an erased anchor hands the claim to its newest surviving witness.
- **Reinforcement is useful recall**: the claims an owner turn was served are logged, and the night scores
  each by the owner's next message: used in the reply and judged right (a checked verdict) reinforces it
  and revives it from cold; restating it never does.

With the flag off nothing new is folded, held or cooled, and the readers keep
reading what was written while it was on. Recovery uses release rollback.

## Wishes

With `faculties.wishes` on (and the mastery drive's weight above 0), each night
the mastery drive may wish for one capability the agent lacks
(`P/mind/wishes.py`, architecture 4.8 item 5). It reads the failure classes of
the last 14 days with at least two *verified* failures or blocked tasks (a
Hermes failure with its reason, a check or the owner; a failure with no reason
verifies nothing), with the lessons already learned for each, and makes at most
one tool-less call in the night's budget. The answer names one class and the
kind of capability it lacks (a tool, an integration, a data source, a skill or
knowledge), and cites at least two failures of that class by the labels it was
shown. A task the owner rated `wrong` (`rate`, which records `verified: owner`)
is a verified failure of its class too.

With `faculties.work_records` on, the owner's own work in the work log
([WORK.md](WORK.md)) feeds classes as well, read without the ledger's writer lock
(`WorkStore.failures`) in a worker thread, off the event loop. A piece of work whose status is `failed` is one
check-verified failure of each class `work:<tool>:<kind>` its failed calls name:
the tool and the host's failure kind (`tool_error`, `timeout`, ...; `failed` when
the call reported none). A failure a later verified call to the same resource
made good is left out, as the work status leaves it out. Work the owner corrected
(the nightly owner-correction overlay) is one owner-verified failure of
`work:<tool>:corrected`, the tool of its last acting call. A guard's block
(`plugin_block`, `guardrail_block`, `scope_block`) is a permission decision, not a
missing capability, and is no class; a cancelled call (`cancellation`) is not a
failure either; unproven or partial work is no failure. A
work class's attempts are the pieces of work that used its tool, and the packet
shows only code's words for it (the tool and the failure kind), never a title,
an argument or a result. The wish cites the work as `work:<root id>`; measuring
it reads the same work log.

A wish is typed. What a yes is for is the capability code names from the kind
and the class's own record, "a data source for '<the class's topic>'"; it has no
free-text field. What a yes allows is a closed set of steps decided by code, not
by the model: research how to gain it; with `faculties.skills` on, a procedure
draft; and, when only a code change could add it, a request note for the
developers. The ask shows that set in full, never shortened, with the sentence
that a yes changes nothing about what the agent may do (no access, permission,
credential, contact, grant, budget or autonomy setting). Every reader (the ask,
the goal's title, description, topic and concern, the concern summaries, the
digest, the autobiography, `/v1/mind/wishes`) names the typed capability.

With `faculties.wishes_named` (on by default) the model may add its own words
for the capability: a name (one phrase, at most 120 characters) and why (at most
300). They are a description under a label, never what a yes gives. Words that
are not printable, too long or more than one phrase (a second sentence, a clause
after a colon or semicolon, a line break; a mark inside a name such as `Node.js`,
`.NET`, `PDF 1.7` or `data.gov` is part of it) are withheld, and the typed wish
stands on its cites; a name or why that is not text is withheld the same way.
With the setting off the answer is the kind and nothing more. In either form the
schema is enforced, not only offered: an answer with any field outside it (the
wish's or the answer's own), a class, kind, cites or `needs_code` left out or of
another type, a kind outside the five or cites it was not shown is refused
whole, never read as a smaller wish.

The text code did not write (the class's topic, and the words when there are
any) goes to one more typed decision in the night's budget, the scope check
(`mind_wish_scope`): given the typed capability and what a yes allows, does the
text ask for more? `within` shows the words; `beyond` refuses the wish before
anyone is asked (it is not asked again that night, and the reason is never put
back to the model); `unsure`, no answer, an answer outside its one enum field or
a spent budget withhold the words and raise the typed wish without them. The
check can only narrow: it never changes a kind, a class or a step. Authority does
not rest on it: whatever the words say, a yes adopts only the typed steps under
ordinary authority, and what the agent may do stays the owner's configuration.

The words appear in exactly three places: the ask's lead, under "In my own words
(a description only)"; `words` in `/v1/mind/wishes` (with `words_check`:
`within`, `unsure`, `unchecked` or `malformed`); and a step's quoted context,
under "The mind's own words for it (data)". Every step's title and instructions
are fixed text. The night counts `wishes_words_withheld` and `wishes_refused`;
the last refusal and why are in `/v1/mind/wishes` (`stats.refused_last`).

A valid wish is one ask (`wish` is an owner question: an ask at every level,
sent as its own message and listed in the digest's "Waiting for your OK" part as
"Let me work on a capability I lack.", the model's words under their label when
they were shown, then "A yes lets me only work toward <the typed capability>:
<the steps>. It changes nothing about what I may do: ..."). Only the owner's own
yes to that ask grants it (Asks); an answer to a question never does
(Questions). A yes adopts an agent-owned goal whose steps are planned,
formed as planned without a deliberation call: research how to gain the
capability; with `faculties.skills` on, a procedure draft run as a reflector of
the wished-for class (its lessons start as candidates and become a skill only
after verified wins); and, when only a code change could add it, a request note
for the developers, shown with the wish and in the digest. The ask names these
steps and records them: the goal takes the ones the owner said yes to and no
more (a flag turned on later adds none; one turned off drops its step). A planned step that
fails is tried once more inside the goal's task budget; one the owner declines
is never formed again, nor one authority dropped, and the goal closes there.
When the goal closes the
wish is measured: the class's verified failures per attempt in the 14 days
after, against the 14 days before it was made. It records `helped` (the rate
fell over at least two attempts), `not_helped`, `no_evidence` (fewer than two
attempts) or `not_pursued` (the goal did no step). A class is wished for at its
worst, so it tends to improve anyway: a declined or unanswered wish is measured
the same way from the moment it was settled, as the control, and `stats.wishes`
reports `helped_rate`, the control's `fell_rate` and their difference, `lift`.

At most one wish a night, none while another waits for the owner, none for a
class whose wish is still pursued or measured; a declined class is not asked
about again for 30 days unless a verified failure of it came after the no (7
days after an ask that expired). A declined work class waits the full 30 days
whatever fails after the no: the owner's work fails daily, so a newer failure
would reopen it the next night. A no is not a verdict on asking: the ranker
learns nothing from it. Turning the flag off withdraws a waiting wish and asks
nothing more; goals the owner already granted run on as ordinary goals.

A wish asked before wishes were typed carries no kind. When the mind starts with
wishes on, before it can take an answer, such a wish still waiting for the owner
is withdrawn ("asked before wishes were typed") with its concern, and its class
may be wished for again, typed, at once. A yes to one grants nothing, even if
that step could not run: the wish is withdrawn instead. One already granted
runs on and is named "a capability for '<topic>'" wherever it is read. Its
goal takes that name at the same start, whether wishes are on or off: the
goal's title, description, topic, concern and plan, the topic, concern and plan
of each step formed from it, and the step still waiting. Affect then files a
failed step under that name, so its note on the next step names the class. The
wish's own record, the body each step was sent with and the steps the owner said
yes to stay as they were.

A wish goal's steps share their topic with the failure class, so affect files a
failed step under that class: a retry with the same plan, on a class that keeps
failing, is first the owner's question (the strategy switch, Feelings).
`colonymind mind wishes` lists every wish with its evidence, state and outcome,
and `/v1/mind/stats` carries the counts and the share that helped. The flag
ships on (on unless shown harmful); those outcomes, and its paired arm, are how
a harmful one would show.

## Outreach to the owner

The social drive turns toward the owner (`P/mind/outreach.py`, architecture
4.10). It speaks up only when it has something concrete: a finding of its own
research that bears on what the owner said they care about (an interest they
declared with `colonymind mind interest`, in conversation, or welcomed), the
owner's own open item after a day or so of quiet (not one due within 48 h:
reminders and heads-ups are duty's), or a named thing the owner said is
stressing them. There is no "anything you need?" message, and a report that
found nothing, or only what was already sent or listed, is never sent as a
finding. An explicit cadence addressed to the owner also supplies a check-in,
including a general check-in when no topic was requested.

Owner cadences use the same `outreach_loop` messages, owner gate, outbox and
reply/silence feedback. They rank as commitment-backed duty, like other
explicit obligations: no learned feedback or interruption-cost penalty, and
no discretionary satiation or overload hold. They still obey quiet hours,
pause, topic mutes, held matters, the minimum outreach gap and `budgets.outreach_per_day`.
Their requested interval replaces silence-driven topic backoff.
`Authority.REQUESTED_TYPES` is unchanged: classifying every `outreach_loop`
as requested would also exempt unsolicited offers from suggest-mode approval.
The first check-in falls due one cadence after the commitment was made; each
later one waits a full cadence after the last send (or attempt if unsent).
An intention still waiting blocks another. Downtime produces one due check-in,
never catch-up messages. Cancelling or completing the commitment stops future
check-ins and withdraws one still waiting to send. The outreach faculty controls
these messages; contact cadence permissions, tiers and check-ins are unchanged.

Each candidate's value is `relevance × novelty × timeliness` and its cost the
interruption (recency of the last outreach, the owner's ignored streak within
the last 7 days, the day's unprompted messages, a "not now" mark on the hour);
the ranker decides.
Quiet hours, a pause, `budgets.outreach_per_day` (3; reminders the owner asked
for are counted apart), two hours after the last outreach, a muted topic and a
topic's backoff hold a candidate before it forms, and at most one unprompted
outreach goes out a tick. An offer of help on an open item waits for a quiet
stretch since the owner last spoke (or since the last such offer), so one quiet
stretch is one check-in. Every
message says why, quoting the owner ("You said \"I care a lot about tidal
energy\", so I looked into tidal energy: ... Say 'dig deeper' for more, or
'not interested' and I will drop it."). What was worth it but not sent is in
the next digest ("Found: ..." in its "Done or found since yesterday" part);
offers that lapsed or were put off are there too ("Offer: ...").

The owner's replies steer it, read in the turn path (`P/mind/reactions.py`):
"dig deeper" rates it useful and starts a follow-up whose answer is sent when
it is ready, outside the daily budget; "not interested" mutes the topic; "not now" pauses for four
hours; "leave me alone today" until tomorrow; "stop checking in" until
"you can check in again" or `colonymind mind outreach on`. A reply that does not
name the topic counts only when it plainly answers the outreach: the owner's
first turn after it, not mid-conversation, and not a request of its own ("can
you find out when the last train leaves?" is a request, not "dig deeper"). A
bare "stop" or "not today" means something about outreach only as such a reply. A reply linked by
its topic that the phrases read nothing in is engagement, unless the decision model reads a class in it
(Instinct's point `outreach_reply`, once its fast tier is live: [DECISIONS.md](DECISIONS.md)). A stop, a pause or
a resume beside someone else's words (a quote, a forwarded chat, a question about what to reply) is Instinct's
point `owner_instruction`: one typed call on the mind's model decides whose instruction it is; with no answer a
stop or a pause stands (the owner can lift it) and a resume counts only as a reply linked to an outreach. Reminders the owner
asked for never stop. Silence for a day counts as a weak "ignored". What is
learned is the verdict and its feedback, the topic's interest level, a mute,
the pause and the hour's mark, and at night the lessons over the rated
messages. With `faculties.outreach` off none of it runs.

A configured [`reply_timing` learner](PLASTICITY-LEARNERS.md#reply-timing-reply_timing)
can hold discretionary outreach for a likelier reply hour within two hours,
after the existing rules and within the message's expiry. It never holds
requested answers, duty, health notices, approval cards or an item due within
two hours. The first hold limits the total delay; a keyed five percent of
would-be holds proceed with their sampling probability recorded. Confirmed
sends receive paired numeric forecasts and an outcome after the full hour;
measured regression rolls the learned revision back automatically. The
plasticity report includes timing, calibration and comparison counts. With no
loaded learner or no qualified revision, timing is unchanged. It reads owner
turn times and channels, with no presence input; the outreach budget stays the
same. The two-hour bound and forced share are plan assumptions.

## The floor

Hands use the same effect table and ask path with `origin: hand`. Recognized
floor calls, including configured publishing destinations and owner account
hosts, wait for the owner's own answer. The ask shows the exact arguments,
working directory and bound target values; a permit runs once and changed
targets require a new ask. The owner's structurally identified conversation
keeps its existing authority, while completion and synthetic turns use the
hand floor. Mind-started hands may use shell and code with hand budgets and
the same floor; shell limits remain best effort. See [Hands authority](HANDS.md#authority).

Four kinds of effect always wait for the owner's yes, at every level:
moving money, irreversible deletion, credential changes and bulk messages to
third parties. What decides is what an action does (the tool and its target),
never how the intention or the task was worded: "settle the invoice" and
"pay the invoice, it is $1200" are the same effect, and a finding that quotes
a price moves nothing.

At the intention level a task's class comes from its toolsets. A toolset
whose effects cannot be bounded per call (`mind.floor.executors`: a shell,
code execution, coding, debugging, a browser, computer use, the desktop
tools, delegation) makes the task floor-class: it asks at every level, and
the owner's yes to it is what lets its commands run. No standing grant makes
executor use the mind's own. With the default worker toolsets no task is
executor-class. A message is classed by its recipient; one concern or one
source reaching more than `mind.floor.bulk_recipients` distinct people other
than the owner in 24 h (default 3) makes the next such message bulk.

At execution every effectful tool call of mind-originated work is classified
by `authority.effect_of`, a table the plugin guard carries byte for byte (a
test compares the two copies):

| Tool | Effect |
|---|---|
| read-only tools, the kanban tools, `colonymind_*` tools | none (each ColonyMind tool enforces its own caller rules) |
| `write_file`, `patch` | none: confined to the task workspace, anything outside is refused |
| messaging tools | a message: the recipient's permission (below) and the bulk count |
| `cronjob_manage` create, update, run | unbounded (a deferred prompt); a job a mind run creates or changes is recorded as mind-origin, by its id |
| `cronjob_manage` remove | irreversible deletion, unless the job (by its id) was made by this task |
| `skill_manage` delete, remove, remove_file; `memory` remove (in a batch too) | irreversible deletion |
| `browser_vault_fill`, or `browser_vault` with `action: fill` | money movement for a payment item (or one whose kind cannot be read); none for a login or an address |
| `browser_vault_save_login`, or `browser_vault` with `action: save_login` | a credential change |
| `terminal`, `execute_code`, the browser, computer-use and desktop tools, `delegate_task` | what the recognizers name in the command, code or call; anything else is unbounded |
| `colonymind_action` (a declared action, below) | the action's class (`none` or a floor class) under the action's own name, with its target and amount from its checked arguments; a call its declaration does not allow is refused as a call to correct |
| a tool `mind.floor.tools` declares (only a tool the table does not read itself: declaring the shell, code, a browser, messaging or memory is a problem, and the table's reading stands) | the declared class (`none` or a floor class, named explicitly), with its amount, currency and target arguments; a declaration with a problem makes each call wait for the owner's yes to exactly it |
| any other effectful tool | unbounded |

Nothing is parsed to set a call free. The recognizers (`FLOOR_PATTERNS`, and
a request to a host in `mind.floor.money_hosts`: by default the public API
hosts of well-known payment providers) only ever hold: a command they name is
that floor effect wherever its paths point, the worker's own workspace
included (`rm -rf ./build` waits like `rm -rf ../x`), and a command they do
not name is an unbounded effect. A shell spells a path, a hook, a redirection
or an appended command in more ways than any reader sees, so no command is
read as the worker's own and no standing grant covers a shell, code, browser
or delegation call. An amount is only ever a declared tool's or action's
checked number argument; a command's text never states one.

The core's `tool_call` wrapper for deferred tools is no call of its own, and
a mind run's wrapper is never sent for a verdict. One whose `calls` the core
cannot read goes back to the worker to be written again. One that reads
passes only when every entry is a read (let through as its direct call is) or
a connector tool (the core brings each to the guard under its own name, where
it is judged); for any other entry the worker is told to call the tool
directly, so the call is judged, and held, under the tool's own name. An ask
formed under the wrapper's name before this, still waiting for the owner, is
withdrawn by the mind at the next tick.

The verdict at execution, in order: the off switch and the deny list; the
call the owner approved exactly (every argument, and the directory a command
runs in; a relative `workdir` whose directory cannot be told is part of the
call as written) acts once; an unbounded effect acts when the owner approved
the task (its intention asked and the owner said yes); a floor effect acts
when a standing grant covers it (each use is recorded). A call of a declared
tool whose declaration has a problem acts only on the owner's yes to exactly
it, whatever the task. Anything else is held. A message then follows the level x
permission table the intention uses: `auto` acts, `ask` asks, `never` is
refused, at every level (there is no exemption at `trusted`; at `suggest` a
worker's send to an `auto` contact asks, as the intention does), and the
owner is always reachable. A plain send the table asks about becomes the
message intention it would have been: the words frozen in the ask are the
words that go after the owner's yes. Any other call the table asks about (a
reaction, a send to a thread, a cron job delivering to a contact on ask) is
held as the exact call.

Bulk is counted per task: the people a task's worker runs, its continuations
and the kanban tasks it made (the plugin reports each one) reached in 24 h,
kept in `mind.db`; a kanban task the sidecar did not dispatch is counted by
its own lineage, and so is a continuation the owner's yes to one of its held
calls made. The outbox's own messages are counted by concern or source
whenever one is queued, a deferred one reconsidered included.

**Holds.** A held call becomes one ask (`floor_effect`, deduplicated by the
digest of the complete call) naming the exact call (tool, arguments, the
directory a command runs in) and the task it came from. The worker is told it
waits and ends its run with a truthful report; the task is marked blocked (a
hold is not a failure: the breaker does not count it) and its worker's report
does not settle it. The owner answers the ask's own message, read strictly
(Asks); an answer after the ask lapsed approves nothing. A yes is a continuation
task carrying the original plan and the exact call to make, which runs once,
within `mind.floor.permit_hours` of the yes; any other call, or the same one
again, is held again. The held task settles with the continuation: its outcome,
its summary and its structured result. A no settles the task denied, and a lapse
settles it expired. Holds and grants are for mind-originated work: the owner's
own conversation is never gated, and a request the owner makes for later that a
worker carries out holds its floor effects, each as its own ask. A payment card
fill (`browser_vault_fill`, or `browser_vault` with `action: fill`, on a payment
item) also asks Hermes' own consent,
which a worker cannot give: the owner's yes to that fill (or their standing grant
for it) is that consent, for that item in that task, once; another item with the
same label and site, or the same item from another task, asks Hermes' consent as
before.

A cron job a mind run creates would otherwise run later as an owner-lane cron
session. When a mind run's create or update is let through, the sidecar keeps
a pending mark; once the call returned, the plugin reports the job's id
(`POST /v1/mind/cron-origin`), so every job form is recorded (a job of skills
alone has no prompt). When a job runs, the plugin reads its id from the run's
task id (`cron:<job id>:<run>`, which a compressed session keeps), its
session (`cron_<job id>_<time>`) or the nearest such ancestor of a delegated
child's session, and asks `GET /v1/mind/cron-origin`: a mind-origin job's run
is a mind run (its effects classified, no file writes: it has no workspace).
Only the sidecar's own answer is kept for the run. While nothing says whose
job it is (the sidecar cannot say or answers 404, a pending mark was never
confirmed for a job made around it, a cron run names no job, or a session's
ancestry cannot be read) the run is gated the same way and asked about again
at its next call. An owner's job is not gated, and neither is any cron run of
a sidecar without the mind routes (it has made no job).

## Standing grants

A standing grant lets one class of floor effect act without asking, through
the structured tools it names, within limits the owner sets and sees:

```yaml
mind:
  grants:
    - id: renew-certs
      class: credential_change        # money_movement | irreversible_deletion | credential_change |
                                      # bulk_third_party_messaging
      scope: renew TLS certificates   # the words shown in asks and state
      tools: [renew_certificates]     # the structured tools it covers (required): here a declared action
      match: [example.org]            # the exact targets it covers (a declared target, a recipient id)
      per_day: 2                      # uses per rolling 24 h
    - id: small-payments
      class: money_movement
      scope: pay invoices up to 50
      tools: [pay_invoice]            # a declared action or a declared tool with an amount argument
      max_amount: 50                  # the most one effect may move; an unknown amount is never covered
      currency: USD
      per_month_amount: 200           # the total per rolling 30 days
      expires: 2030-12-31             # past it the grant is inert
```

A grant names structured tools only: a declared action, a declared tool
(`mind.floor.tools`), a messaging tool or `outbox` (bulk), `cronjob_manage`,
`skill_manage` or `memory` (removals), `browser_vault_fill` or
`browser_vault_save_login`. Those two names hold on every Hermes: where
Hermes has the one vault tool `browser_vault`, a call is read as the older
tool of its `action`, and `browser_vault` itself is never a grant's name. It
never names a shell, code, browser or
delegation tool or toolset, and never matches command or code text: such a
grant is malformed and inert, and each of those calls outside an approved
task stays an ask for the owner's yes. A call with a problem (an action its
declaration does not allow, a declaration with a problem) is never covered.
A grant is the owner's to give in their own conversation, or the operator's to
configure, and no request gives one by itself. In the owner's conversation,
`colonymind_self grant` with the grant and the owner's words files a
`grant_confirm` ask showing the grant in full; it is active only on the owner's
own yes to that ask in their chat, read strictly (Asks), and a guest, a cron run
or a worker is refused. The operator puts it in `mind.grants` in
`colonymind.yaml` (a mind run can neither write outside its workspace nor name
the file). `colonymind mind grants add` adds nothing: it checks the grant and
prints the `mind.grants` entry to put there, and `POST /v1/mind/grants` naming
no conversation is refused 403 `approve_required` for every credential (Asks:
console and API answers). An approval of a grant undone within 10 minutes
withdraws it.
`revoke` (the CLI, `colonymind_self revoke <id>`) is free and immediate; a
revoked configured grant stays revoked until its configuration changes, and
its id stays its own: a grant added under it is refused (one confirmed before
the configuration took the id is shown inactive, with the reason). A malformed
grant (limits are finite numbers) is inert and shown as invalid; it never
widens anything.
`colonymind_self state`, `GET /v1/mind/state` and `colonymind mind grants` show
every grant with its source (config, the ask's code, or cli for one an earlier
release added from the console), its limits, its
uses and amounts in the current windows, its expiry and its validity. The
configured set is fingerprinted at each start; a change since the last start
is shown in state and told to the owner once ("standing grants changed:
added ..., removed ...").

### Declared actions

A recurring action the owner wants done without asking, which today would be
a shell command, is declared as a small dedicated tool: a fixed command whose
only free words are checked arguments. The worked example renews one site's
certificate:

```yaml
mind:
  actions:
    renew_certificates:
      description: Renew the TLS certificate of one site
      class: credential_change        # none or a floor class: what running it does
      command: [certbot, renew, --cert-name, "{site}", --non-interactive]
      args:
        site: {type: string, enum: [example.org, www.example.org]}
      target: site                    # the argument a grant's match reads
      timeout: 300                    # seconds (default 120, at most 3600)
  grants:
    - {id: renew-certs, class: credential_change, scope: renew TLS certificates,
       tools: [renew_certificates], per_day: 2}
```

The plugin offers `colonymind_action` once an action is declared; the model
calls it with the action's name and its arguments (`{"action":
"renew_certificates", "args": {"site": "example.org"}}`). The command runs
without a shell: its words are passed as they are written, and a word written
`{name}` is replaced by that argument once it passed its check. An argument is
`type: string` (with an `enum` of the values it may take or a `pattern` its
value must match whole; it is one line and never starts with `-`, so it cannot
add an option), `integer` or `number` (with `min` and `max`); `default` makes
one optional. `amount`, `currency`, `default_currency` and `target` name the
action's arguments as a declared tool's do, so a payment action's amount is a
checked number a grant's `max_amount` and `per_month_amount` can hold (a
payment example: `command: [pay-invoice, --invoice, "{invoice}", --amount,
"{amount}"]` with `invoice: {type: string, pattern: "INV-[0-9]{1,8}"}` and
`amount: {type: number, min: 0, max: 1000}`). A call its declaration does not
allow (an unknown action, an argument missing, unknown or refused by its
check) is refused before anything runs, as an argument the model can correct.
In the owner's own conversation (and their own cron jobs) an action runs as they
ask; in a mind run it runs when the guard lets it: its class `none` always, a
floor class on a grant that names it or on the owner's yes to exactly that
call. A guest never runs one. An action whose declaration has a problem is
shown with it in `GET /v1/mind/state` (`actions`) and runs nothing; an action
never takes the name of a tool the table reads itself. The
declaration is read again at each call, so a changed or removed action is what
counts; in a mind run it runs only when the sidecar allowed that same reading
of the call, so until the sidecar restarts on a changed declaration the call
waits for the owner.

### Standing allows

An ask whose kind of work can stop asking offers "always" in its own message, with exactly what it would allow ("Reply
yes or no, or 'always' to let me contact Sam without asking first."). Three kinds are offered, at `standard` and
`trusted` only:

- **contact**: an ask to reach a contact whose permission is `ask`; the allow sets it to `auto`, only while it is
  still what the ask showed (an opt-out since wins);
- **toolset**: a task that needs extra tools; tasks with exactly those tools then act without asking, and no shell,
  code or browser call inside them runs on the allow;
- **grant**: a held call of a structured tool that moves money, deletes, changes a credential or messages many
  people; the allow is a standing grant for exactly that class, tool and target (and currency), at most the amount
  the ask showed, once a day. Asks to pay the same payee are one kind whatever else the call carries, and a grant
  allow counts each call its grant lets run as a use.

It is never offered at `suggest`, never for a shell, code or browser call, never on a task whose tools include a
terminal, code execution, a browser or delegation (nothing confines what those reach to the task's own folder, so
such a task keeps its one ask, and the owner's yes to it is what lets its commands run), and never on an ask about
who a matched name belongs to. The owner's own "always" is read like any answer that widens what the agent may do
(Asks: Reading the words): plainly, with no condition or question, on the ask's own message. It approves the ask and
adds the allow; a plain yes approves the ask alone. Nothing but that answer adds an allow, and no allow lifts the off
switch, the deny list, the circuit breaker or the budgets.

`colonymind mind allows` (`GET /v1/mind/allows`, the owner's view) lists each allow with the words the owner agreed
to, the ask it came from, when, and how often it let work go without asking. Three ways take one back at once, and
the asking comes back (a contact goes back to `ask` unless their permission changed since; a grant allow's grant is
withdrawn): `colonymind mind allows revoke <id>` (`POST /v1/mind/allows/{id}/revoke`, a narrowing any credential may
make); the owner's own words in their chat ("ask me again about Sam", "always ask me before paying acme", "ask me
again about everything"), where a negation, a question or someone else's words take nothing back; and an undo
within 10 minutes of the "always". A revocation takes effect at once even when the contact store cannot take the
permission's reset yet: the agent asks before contacting that person, retries the reset at each tick until it lands
(or the permission changed since, which is left as it is), and the listing shows it under `resetting`; a new
"always" for that person meanwhile keeps their `auto` and drops the reset.

One line of the daily digest, at most once in 7 days and only while an allow is in force, says what goes without
asking and how to bring the ask back, and offers an allow that let nothing go for 90 days for removal; the line
never makes a digest on its own. After `mind.asks.propose_after` (3; 0 turns the offers off) of the owner's own
yeses in a row within 30 days to asks of one kind, the agent asks once, as its own message: "Stop asking before I
contact Sam? You said yes the last 3 times I asked." A yes adds the allow; a no or no answer holds the next offer
back for 30 days. A no or an undo in between starts the count again; a console answer and a lapse count neither
way.

## Asks

An ask is an intention the mind will not carry out without the owner's yes: an ordinary one (a message to a contact
whose permission is `ask`, a task the level asks about, a private value's use in one call: docs/SECRETS.md), a floor
one (an executor task, a held call: The floor), an owner question (an identity link, a check-in matched by name
only, a change to who gets messaged, a wish) or a standing grant to confirm (Standing grants). It lives only in the
sidecar: nothing is created in Hermes until the owner approves, so no board action can approve one. Each ask has a
short code (`K7F`), its id for the console and the API; no ask message carries it (a private value's use is the one
exception: docs/KNOWN-GAPS.md), and a code typed in chat approves nothing (Which ask a message answers). A question
the mind asks the owner (which of two statements to keep) is never an ask: it has no code, and where the level would
make it one it goes to the digest instead (Questions).

`suggest` holds the mind's initiative, not what the owner asked to be told: a reminder or heads-up the owner asked
for, the question about a recipient the owner named that the store cannot resolve, and the mind's own health and
breaker notices go out at every level but `off`. The learned multiplier never weighs a commitment someone made, and
a blocked task holds no `concurrent_tasks` slot. A name-only identity link ("Is sam@example.org on email Sam?") is
always an ask; the answer links or rejects the handle in the contact store, and a `no` is not a verdict on asking. A
yes folds a shadow that held the handle into the contact, and never an established contact: that link is refused
and the owner merges the two explicitly if they are one person.

### The ask's own message

Each open ask reaches the owner as its own message, in plain words, with no code, no internal id and no expiry line:
what the agent wants to do with its object named, then the reply line, "Reply yes or no." An ask whose yes lets
something happen that cannot be taken back says so first ("A payment cannot be taken back once it is made.", "What
is deleted cannot be recovered.", "A credential once changed cannot be changed back by me.", "Messages cannot be
unsent once they go."). A message ask shows the exact words a yes sends, in full, and a use of a private value where
it goes. A wish shows what a yes allows, in full: the typed capability, the steps a yes adopts and that it changes
nothing about what the agent may do (Wishes); only its lead, the mind's own words first, may be shortened. What a
yes authorizes is never shortened: past 1,500 characters it goes to the owner whole in its own message
(`ask_in_full`), and a yes counts only when that message was delivered before the owner wrote the yes (The daily
digest).

- **At most `mind.asks.messages_per_day` (3) a day.** An ask message counts for 24 hours from when it reached the
  owner, and one still on its way counts whatever its age. Asks due beyond that go together in one numbered list, at
  most one list a day and 5 asks in it ("2 more things wait for your OK: 1. ... 2. ... Reply 1, 2, both or
  neither."); its numbers mean that list, for a day. A floor or widening ask (Reading the words) always goes on its
  own and is not counted.
- **One reminder.** An unanswered ask is sent again once: right after the first daily digest delivered at least 12
  hours after its first message, or `mind.asks.remind_hours` (24) after that message when no digest comes first.
  Never a second. "Later" moves it to after the next digest.
- **The level.** At `autonomy: suggest` an ordinary ask's first message waits for the next digest; a floor ask goes
  at once at every level, at the owner gate's `high` urgency.
- **A pause first.** With the owner gate on (the default), an ask's message, its reminder and a list wait while the
  last message in the owner's chat is the agent's own, under `mind.asks.quiet_minutes` (10) old and not answered yet:
  until the owner writes or the minutes pass. A floor ask waits at most those minutes, any other at most the longer of
  those minutes and the gate's `max_wait_minutes` (60), and two new asks never arrive back to back into one "yes".
  A floor ask for work started by the owner's request goes after the reply to that request without the conversational
  pause on every platform, including WhatsApp, SMS, RCS and API. Its stored note has origin `hand`, and its
  `held_from` names the hand. The controller writes that hand's task record. The task must have origin `owner_turn`,
  with a recorded `origin_ref` identifying the turn by platform, chat or session, and message id. The recorded
  turn's sender must be the owner contact. Claims in note or notice text do not establish this link. `delivery_ref`
  only selects the RCS owner notice delivery path; WhatsApp and SMS notes do not need it for this exception.
  It still waits behind another ask or message the gate let go that is younger than the pause and unanswered.
  Background asks keep the full pause and bounds, as do hands started by the mind or by a guest's turn. Who spoke
  last and when, and the validated link to the request, decide, never the words (docs/OWNER-GATE.md).
  A shown ask or a plain question about an answer with `answer_turn`, and the line saying what an answer did, answer
  the owner and do not wait for the pause. The body still holds ask notices until the target chat's turn replies and
  for 5 seconds after it; a turn with no reply after 10 minutes holds nothing.
  A hand's notice also waits for its parent completion review to end, even when
  the hand ran other tools first. This wait and gap end at the earlier of
  3 minutes after the hold and the hand's wall deadline.
- **Where the owner is.** An ask's message, its reminder, a list and an ask's full text go to the chat of the owner's
  most recent message the answer route read in the last hour, else to their primary handle; a chat where such a
  message failed in the last day is not tried again that day. What answers the owner's own message (a shown ask, a
  plain question, what an answer did) goes to the chat of that message.
- **Silence is not a no.** At `mind.ask_expires_hours` (72) an ask lapses: nothing is done, its messages expire with
  it, a held task settles `expired` with it, and it is no verdict. An answer after the lapse approves nothing.

### Who may answer

A message is read as a possible answer only when all of these hold; otherwise it is not read at all:

- the request's credential is granted the owner on the message's platform (docs/CREDENTIALS.md): an integration's
  credential never reaches the reader;
- the sender resolves to the owner contact, in the owner's direct chat with the agent: never a group, a guest's
  chat, a scheduled run, a kanban worker or a delegated session;
- the channel does not mark the message as the agent's own account's (the WhatsApp bridge's owner echo) or as
  forwarded;
- the words are the owner's own: the turn's message after the secret seal, with the gateway's prefixes and quote
  pointer removed, or a voice note's transcript. Never the model's text, a tool result, a skill, a memory or a web
  page: the read runs before the model does;
- the message has not been read before. Each is read once, keyed by its platform, its chat and its id there (else
  its turn id), whichever route brings it first (`answer_turns` in `mind.db`: what a read did, never the words, kept
  30 days).

The Hermes plugin reads the owner's message before the model runs (`POST /v1/mind/answers`,
docs/HERMES-ADAPTER.md), within 1.5 seconds in all, when an ask is open or the message replies or reacts to
something. A message it could not read in time is read from the captured turn instead, without the chat's ledger:
then only a reaction or a quoted reply binds it, never the next-message rule. A message over 8,000 characters is
never read from part of it: the captured turn is read whole. A voice gateway's spoken turn is not read as an answer
at all (it names no message of the chat), so a yes said on a call answers no ask: the ask waits in the chat.

### Which ask a message answers

By structure, judged at the moment the message arrived; the first rule that applies decides (`answers.bind`):

1. **A reaction on an ask's own message**, matched by its delivery id, its platform and its chat: 👍 ✅ 👌 ✔ is
   yes, 👎 ❌ 🚫 no, any other emoji nothing. A reaction on anything else, a list included, is ignored. A tap that
   answers an ask takes no model turn.
2. **A quoted reply** to an ask's own message binds to that ask; to a list, to that list, for 24 hours. A quoted
   reply to anything else (the agent's message, a closed ask's) binds nothing, whatever the chat showed last.
3. **The next message.** It binds to the ask (or the list) whose own message was the last thing in the chat when
   the message arrived, as the plugin's ledger of that chat shows it: the body's sends, the agent's replies and
   interim messages, its own sends to the owner, every scheduled run's reply (in every owner chat: the plugin cannot
   see where a scheduled run delivers) and a turn still under way. A "yes" after any of them is conversation. An ask
   delivered after the message arrived, or less than 3 seconds before, is never what it answered, and two sends
   within 5 seconds of each other in no order the plugin fixed bind nothing. A floor or widening ask binds this way
   only within 12 hours of its own message, a list within 24 hours.

Nothing else binds. A yes or a no whose words name exactly one open ask (its contact's name, its file or folder, its
payee, or the code the console shows) puts that ask in front of the owner again as its own message; it never
approves it. A code
is no reference: typed beside a yes that binds by the next-message rule, it is a plain yes. A clean yes or no that
binds to nothing while an ask is open is counted (The asks report). The reader queues the most recently shown
eligible open ask in that chat through the existing show path, even with older open asks beside it.
A floor or widening ask must have been shown within `FOLLOW_UP_HOURS` (12 hours) and must not have expired.
The unbound message approves and declines nothing. The model is told which ask is queued again in this chat after
its reply, for a fresh yes or no. It must not create a replacement hand or send the owner elsewhere to answer.
The repeated ask follows the recorded answer's chat, including its native RCS delivery reference, even when the
owner later writes in another chat. See [APPROVALS.md](APPROVALS.md).

### Reading the words

Only once a message is bound are its words read, by fixed tables (`colonymind/mind/answers.py`). This is the one
place in ColonyMind where words decide, and only the owner's own words, after a structural binding. Someone else's
words the owner passes on (a quotation, "Sam said yes", a line naming its speaker, a pasted page) are blanked first.

| Reading | The owner's words | What it does |
|---|---|---|
| yes | yes, y, yeah, yep, sure, ok, okay, fine, go ahead, go for it, do it, please do, sounds good, approve, confirm, send it, ship it, of course, that works, 👍 ✅ 👌, each also with "thanks", "please" or a glad emoji; "k" and "go" alone only for an ordinary ask | approves |
| no | no, nope, nah, don't, no thanks, skip it, cancel, leave it, never mind, deny, reject, decline, "don't do it", 👎 ❌ 🚫 | declines |
| later | later, not now, not yet, hold off, ask me later, remind me later, tomorrow, not today, in a bit | the ask stays open with no verdict; its reminder moves to after the next digest |
| numbers, all, none | against a list: "2", "1 and 3", "yes 2", "no 1", all, both, none, neither | each ask it names, yes or no |
| unclear | an answer word with a hedge anywhere ("but", "except", "if", "only if", "unless", "instead", "wait", "hold on", "maybe", "first"), a question in the first sentence, a negation within three words of the yes ("not sure"), a yes and a no in one message, or words beside the answer the tables cannot place | one plain question, as its own message: "Just to check: <the ask>? Yes or no.", or over a list "Which one? 1. ... 2. ... Reply 1, 2, both or neither." |
| none | no answer word first in the owner's own words | conversation: the agent answers it as usual |

A yes followed by a separate request ("yes, and remind me tomorrow") or by a question in a later sentence ("Yes. How
long will it take?") is a yes to an ordinary ask; the rest reaches the agent as usual. One plain question at most
goes per ask in 24 hours; still unclear after it, nothing is done and the ask stays open. "Always" ("yes always",
"yes, don't ask again") is read only where the ask's own message offers it (Standing grants: Standing allows), and as
strictly as a widening ask: elsewhere "yes, always" reads as unclear. Where the tables read an answer to one ordinary
ask as unclear, Instinct's `ask_answer` point may read it first ([DECISIONS.md](DECISIONS.md), `ask_answer`): it ships
in shadow, and once live a model's no settles no, while a model's yes only asks back ("Doing this: <the ask>, OK?")
until that kind of ask has earned it.

**A floor or widening ask is read strictly.** A floor ask (money, an irreversible deletion, a credential change,
bulk messages, an executor task, a held call) and one whose yes widens who gets messaged, who is who or what may run
without asking (a people change, an identity link, a check-in matched by name only, a standing grant to confirm):
"k" or "go" alone, a question anywhere in the message, or anything beside the yes (a separate request included)
does not approve it. A list's yes never approves it: it is shown on its own instead, and a list's no declines it. It
binds by the next message only within 12 hours. A wish is an ordinary ask: its yes gives only the typed steps
(Wishes).

### What an answer does

- **A yes or a no** settles the ask through its own path: a task is approved (for a held call, a continuation that
  makes that call once), a message goes, a grant becomes active, a people change or a link applies, a wish is
  adopted. The settling history row records `answer: {path, reader, message, by}`: the path (`reaction`, `reply`,
  `follow_up`, `list`; `console` from the CLI or the API), the reader (`tables`, or `decision` for `ask_answer`), the
  message's key, never its words.
- **The agent is told what the message did** before it replies (what was approved, declined or put off, the
  question that goes, or that the message answered no open ask and which asks are open), so it acknowledges once and
  never asks again. It has no tool that approves: `colonymind_self asks` lists the open asks in words and
  `colonymind_self show <n>` sends one to the owner again, in the sidecar's own words.
- **A tap, or a message read only after the agent replied,** gets one line saying what it did ("OK, approved: ...",
  "OK, declined: ...").
- **An ask notice, a shown ask and a plain question** go as their own message, at least 5 seconds after the reply
  of a turn in the target chat, so they land last and apart. A turn with no reply after 10 minutes holds nothing.
  A background ask needs no extra wait when that chat has no open turn or reply still settling.
  A hand ask waits for the parent to finish its completion review, regardless of earlier tools,
  plus the same reply gap. The controller caps that wait at 3 minutes after the hold and the hand's wall deadline.
  Review settlement plus the gap, or the cap, hands the notice directly to delivery without a second chat wait.
  A verified owner request link stays with the ask across controller cleanup and restart; the captured source
  must still belong to the owner on every evaluation.
  A native hand completion that only reports a trusted approval hold settles silently before any model or
  streaming reply. It adds no turn or reply to the answer ledger, so it cannot displace its ask. This applies
  equally to RCS, WhatsApp and SMS and leaves the owner-started ask pause exemption intact.
- **Undo.** "Undo", "cancel that", "wait, no" or "I didn't mean that" (the whole message, polite words aside; a
  question or a quoted line undoes nothing) within 10 minutes of an approval, in the same chat, or a reaction that
  approved changed to a no or removed in that time, takes back the latest approval there whose message arrived
  before the undo's own: work not started yet is cancelled (a permit to use a sealed value not used yet included), a
  standing grant is withdrawn, the standing allow an "always" added is revoked, and a permission or check-in cadence
  a people change set is set back while the
  record still holds the value the yes set. Nothing else a yes did is taken back: a check-in matched by name only
  keeps the cadence its yes set (`colonymind people cadence <who> off` ends it at once, docs/RELATIONSHIPS.md), and
  an identity link, a merge and a granted wish stay. Anything left, work already under way included, is said
  plainly. An undo is no verdict. "I didn't say yes" (or "I never said yes") counts as undo words; where a model read
  the answer (`ask_answer`), "undo" also takes back its no, and "I didn't say yes" or "I didn't say no" take its
  reading back for a day ([DECISIONS.md](DECISIONS.md), `ask_answer`).

### Console and API answers

A request that cites no owner turn (the CLI, a script with the key, any credential) may narrow and never widen:

- `POST /v1/mind/asks/<code>/yes` answers 403 `approve_required` for every credential, the instance key included,
  and says how to reach the owner; `colonymind mind yes <code>` approves nothing and says the same.
- `colonymind mind show <code>` (`POST /v1/mind/asks/<id>/show`, by its code or its id) sends the ask to the owner's
  chat again as its own message, where their answer counts; one already waiting to go is not doubled.
- `colonymind mind no <code>` (`POST /v1/mind/asks/<code>/no`) declines it, recorded `by: operator`, never as the
  owner's own answer.
- A standing grant added with no conversation, and a raised permission, a new or faster check-in or a merge that
  gives a record's handles more reach through the people routes, are refused 403 `approve_required` (Standing
  grants; docs/RELATIONSHIPS.md). A decline, a revoked grant and a narrowing apply at once, and the level, the
  configuration and the off switch stay the operator's.

No credential approves: a secret the owner could mint from a terminal on the agent's machine, the agent could mint
too, so it would add a ceremony and no boundary.

**What stays open.** Whoever can read the instance key can still forge an owner turn to `/v1/mind/answers`, as to
any turn route. Not every approval reaches the owner's chat: a reaction, and a message read from a captured turn,
get the one line saying what they did (`ask_ack`), but a text answer on `/v1/mind/answers` tells only its caller,
for the agent to acknowledge in its reply, so a forged text answer can approve an ask with nothing in the owner's
chat. In every session, the owner's own included, a tool call that reads `api.key` or names an answer route is noted
in the audit log (`answer_forgery_attempt`) and told to the owner at most once a day. In the owner's own lane the call
runs (an approval never stands in the owner's way); in a non-owner session and a mind run it is refused as well, and
there a read that reaches the key file however it is spelled counts too. A rule on how a call spells a name is no
boundary on its own, since a shell spells it in more ways than any rule sees: what keeps the key from a contact is that
a non-owner session may call only the guest tools (no shell, code, file-writing, browser or desktop tool, none the
plugin does not know), and the files they take are only those the sender's own messages carried, unless the operator
opts its platform in (docs/HERMES-ADAPTER.md, Owner and guest); a mind run's toolsets have no shell or code tool. `colonymind doctor` warns, with the remedy, when the agent could approve its own asks, and names the
non-owner lanes that could run this machine:

| Check | When | Remedy |
|---|---|---|
| `owner-lane-tools` | an owner lane's terminal, code or file tools run on the machine that holds the instance key | run them on a container or remote backend (`terminal.backend` in Hermes' `config.yaml`), or remove those tools from the owner's platforms |
| `cli-owner-platform` | the terminal (`cli`) is an owner platform of a credential, and an owner lane runs a shell on that machine | `colonymind credentials set <name>` with its other platforms and without `cli` (the doctor prints each command) |
| `whatsapp-mode` | the WhatsApp bridge runs in shared-account mode: the owner's account is the agent's | bot mode on the agent's own number (`WHATSAPP_MODE=bot` in Hermes' `.env`, then `hermes whatsapp`) |
| `guest-lane-tools` | a platform Hermes serves offers terminal, code, file-writing, browser or desktop tools on the machine that holds the instance key, and the operator opted it in (`guests.executor_platforms`), so its non-owner sessions keep them, or Hermes could not list every platform it serves and an opted-in one is missing from the list; a platform not opted in passes, named, since the guard refuses those tools there (it skips, naming them, when the list is partial) | remove the platform from `guests.executor_platforms`, or run Hermes' tools on a container or remote backend |

docs/KNOWN-GAPS.md lists what the approvals design asks for that is not built yet.

### Whose answer teaches

What an ask teaches (the ranker's feedback on that kind of work, and the agent's own feelings, which hold optional
work back after dismissals) comes only from the owner's own answer in their chat: a reaction, a quoted reply, their
next message or a list. An answer given anywhere else (the CLI, a
credential, a script, whatever `by` it names) settles the ask and teaches nothing, and so does a lapse. A no to an
identity link or a wish is not a verdict on asking either. The upgrade to 1.12 clears, once, the verdicts earlier
releases took from an answer the owner did not give in their own chat, withdraws the feedback they fed and logs each
ask's id; a rating the owner gave afterwards stays.

### The asks report

`colonymind mind asks-report --since 7d` (text, or `--json`) and `GET /v1/mind/asks/stats?since_hours=168` (the
owner's view) show how the asks formed in a window went: by type and class; when each first reached the owner; who
settled it (`owner`, `operator`, `lapse`, `mind`) and how (the path and the reader); formation to first sight and
first sight to the owner's answer (median and 90th percentile); shows, plain questions, laters, undos, lapses, and
how long the pause held asks; operator answers learned as verdicts; task-hours of work that waited on the owner;
clean yes or no messages that bound to no open ask; and invented code lines removed from replies, which the sidecar
does not receive yet, so that line's 0 is not a measurement (docs/KNOWN-GAPS.md). The text (the command's output,
the JSON's `text`) is counts and times only, never an ask's words or code. The JSON also lists each ask formed in
the window (`asks`: its `id` and `code`, its type, class, kind and status, when it formed, first reached the owner
and settled, who settled it, the answer, the path and the reader), never its words: share the text where the asks'
ids and codes should not go. An ask settled before 1.12 has the path `unknown` (`console` when the CLI settled it).

### Asks from before 1.12

An ask still open at the upgrade keeps its code as its id. When a notice of the 1.11 format showed it (one message
listing the open asks by code), the ask is sent once more as its own plain message, outside the day's count, and
that is the message to answer. The notice itself binds a quoted reply as a list for 24 hours after it was sent
("2", "both", "none"), as any list does; a reaction on it is ignored. A 1.11 notice that listed one ask is that
ask's own message: a reaction or a quoted reply there answers it.

## Questions

The nightly consolidation raises a question when two statements a person made
about one thing cannot both stand (docs/CONSOLIDATION.md, Contradictions). The
question is written in plain words from the statements themselves, never from
the names the memory files them under: who said them (the owner, or a contact
by name), what each said and when, and how to answer ("You told me two things
that may not fit together. On September 14 you said: "..." On September 15 you
said: "..." Should I keep the first, the second, both, or neither?"). A
statement too long to quote whole is shortened whole or by a checked typed call
(The daily digest); failing both, the question waits for a later night.

**One question at a time** (`P/mind/questions.py`). At most one question is on
its way to the owner or waiting for an answer. Where the level lets the mind
message the owner it goes on its own, labelled like the digest's part
("Question (just answer in words): ..."); where the level would ask it goes to
the next digest's Question part instead (a question never gets a code). Any
other question waits behind it and goes to the digest once the one before it is
answered, found moot or expired. The owner gate may hold a question for the
digest like any message.

**Only a confirmed delivery makes it answerable**: the body's `sent` for the
question itself, for its one clarification, or for a digest that lists it. A
send that failed or ended uncertain leaves it unseen, and it goes to the next
digest. For a day after each confirmed delivery (`ANSWER_WINDOW`), the owner's
turns that came after it are read as possible answers; no other turn is.

**The owner answers in words.** Each such turn is read once, globally: before it
is read, the turn and the question it is read against are recorded
(`question_turns`), so a retried turn is never read again, against this
question or the next. The reading is one typed decision on the mind's router
(`mind_question_answer`: `first`, `second`, `both`, `neither`, `unclear` or
`none`) that also sees the last few turns of that conversation, so a reply to
something else is `none`. It runs off the turn's path (after the turn is
recorded, never delaying the reply), within the turn deadline, and its tokens
are charged to the day's budget. An answer that resolves the question is
claimed atomically (`question_claims`) and revalidated against the exact pair of
statements the question asks about: a second, concurrent or stale answer changes
nothing. `first` keeps the first statement and retracts the second, `second` the
reverse, `neither` retracts both, `both` records them as compatible so they are
never asked about again. A value retracted goes with every live copy of it (the
same value for the same period, said again), not only the one the question
quoted, so no earlier copy takes its place; each change is in the learning log
(stage `owner_answer`). The question closes only once that change succeeded; a change
that failed leaves it answerable and is applied again at the next tick, and a
claim a crash left unapplied (still held two minutes on) is recovered at the
next tick: finished with its answer when the change already happened, applied
otherwise.
`unclear` asks once more in plain words (a `question_again` notice with the
whole question, at most once per question). A question and its clarification
quote the two statements, so each goes out only while they still stand: at
every tick and again when the body pulls it, one whose statements were
forgotten, retracted or found compatible is withdrawn unsent, and so is a
clarification whose question was answered, expired or is no longer answerable.
`none` or no decision (no router,
the budget spent, a timeout, `unsure`) changes nothing. A turn the answer route
read as an answer to an ask (it approved, declined, undid or put off an ask,
showed one again or was asked about one) is that ask's and is never read for a
question, nor is a turn that carries an open ask's code, nor a scheduled or
worker run. An answer in words decides only which statements the memory keeps:
it never grants a wish, confirms a link or a cadence, changes who may be
contacted or approves anything; those are asks, answered through their own
message (Asks).

**Never a reminder.** Sent and unanswered for a day, a question is listed in the
next digest once; listed and unanswered for a day, it expires quietly (and
seven days after its first delivery, whatever happened; a question waiting
behind another or never delivered does not expire); once its statements agree
again (a correction, a retraction) or one of them is forgotten, it closes. A digest's delivery counts from
when the body delivered it, even when it is recovered later. While it is open, an owner turn
about its subject carries it in the context (`Open questions`), saying that the
owner answers in words and that there is no code for it. An older release's
contradiction ask is rebuilt in this form at the next tick and loses its code
(one whose statements can no longer be quoted whole, or agree, is withdrawn), and
a question delivered under the earlier state format keeps its deliveries.
The state is on the question's message row (`result_metadata.question`).

## The daily digest

One message a day, after `mind.digest_hour` (local, outside quiet hours), and
only when something waits or happened (`P/mind/digest.py`). It has up to three
labelled parts, in plain words:

- **Waiting for your OK**: one line per open ask, what it is and when it was
  asked ("Asked 5 hours ago."), with no code and no reply instruction: the ask's
  own message carries the reply line, and a yes binds to that message, never to
  this line (Asks). A question is never listed here.
- **Question (just answer in words)**: the one question waiting for the digest
  (Questions), with no code.
- **Done or found since yesterday**: the tasks and goals that finished, could
  not finish (with the reason given) or started; messages sent to contacts;
  what outreach found or offered; what the owner gate held for the digest; the
  wishes' outcomes; the identity check's runs in which something moved;
  opt-outs; deliveries that may not have arrived (never resent); tripped
  breakers; a paused outreach line. The goals the mind keeps working on ride
  along, with their progress as a sentence of its own, but never make a digest
  on their own.

The recorder adds one line of counts to the third part, and a day with recorded
lines has a digest ([RECORDER.md](RECORDER.md#the-daily-line)).

Nothing that already reached the owner is listed again: a question, a notice or
an answer the owner asked for went out on its own. Every item is one or a few
whole sentences. A text that does not fit its room keeps its leading whole
sentences; when not even the first fits, a typed composition call
(`mind_plain_summary`) writes a shorter version, which must pass the no-cut
check (at most the room, ending as a sentence ends, no ellipsis, and not the
source's own opening stopped short); failing that, the item says what it is and
that the owner can ask for it in full. One digest makes at most 6 such calls,
within 30 seconds in all, each charged to the day's token budget; past that its
items are pointed to. A title a store cut at a fixed length is never shown as it
is (a goal's is written from its full description). Internal field names and
ids never appear: a contact is named by name, else by a handle the owner would
recognise, else as "a contact".

What a yes authorizes is never shortened: an ask's exact message words, a
private value's tool and destination, a held call, a standing grant's terms are
shown in full. Past 1,500 characters they cannot be shown in a digest line or an
ask's message: the owner gets the whole ask in its own message (`ask_in_full`),
sent again whenever the last one can no longer arrive, and a yes counts only
when one was delivered before the owner wrote it. The delivery is read from that
message's own record (the body's `sent`), at startup, at every tick and when the
yes comes, so a delivery whose hook did not run (it failed, or the process
stopped in between) still counts.

The digest is queued once per local date (its dedup key `digest:<date>`) and
expires at the end of that date, and there is one per delivery day: an earlier
digest still waiting or mid-send defers the next, one that arrived today (even
late, for yesterday) was today's, and one whose delivery is uncertain counts for
the day its send began. A restart, a second tick or a second body finds it and
builds no other, the body claims it once (a second claim, from any body or channel, is
refused with 409), and a send the body left unsettled is `uncertain` and never
resent. The digest's row carries its own manifest (`items`: the questions, held
items and findings it lists), written with it, and what it listed is recovered
from that manifest: its items count as listed only once the body reports the
digest delivered, and a digest that expires, fails or ends uncertain hands them
back to the next one.

## The off switch

`colonymind mind off`, `/mind off` in chat, `POST /v1/mind/off` or
`mind.enabled: false` all mean "no further effects". At once, in the sidecar:
no deliberation, the dispatch queue and outbox return nothing, unsent messages
are cancelled and every guard check from a mind-originated run returns block.
At the next plugin tick the body archives unstarted `mind:*` tasks. Running
workers end at their runtime limit. The switch is a marker file
(`mind.off` in the instance directory) plus an in-memory flag, so it works
with the model endpoint down and survives a restart. `colonymind mind on`
turns it back on. `autonomy: off` (`colonymind mind level off`) has the same
effect on effects: nothing is dispatched or sent, unsent messages are
cancelled and mind-originated runs are blocked; what was approved waits for
the level to come back.

A standing "leave X alone" is the deny list (`mind.deny`: tool names, text
patterns and shell commands), enforced by the mind's authority on every
intention and mirrored into the worker profile's `approvals.deny`. A global
pause is the off switch plus stock `hermes pause`; the body holds dispatch
and sends while Hermes is paused.

## Reports

The policy for the mind's faculties is that every faculty is on unless it has shown harm. Two reports make that
policy checkable by a person: a live report of what each faculty did over a window, and a plain-language reading
of the paired benchmark. Both are sentences, not fields: no internal ids or field names, people by role (the
owner, a contact), and each faculty's section names the one setting that switches it off. Both render as
Markdown (the default) or a standalone HTML page (`--format html`), printed unless `--out PATH` names a file.

### What each faculty did

```sh
colonymind report faculties [--since 7d] [--format md|html] [--out PATH] [--bench DIR]
colonymind --instance /path/to/instance report faculties --since 24h
```

`--since` takes a span back from now (`30m`, `24h`, `7d`, `2w`) or a date or time (`2026-09-01`). Daily
retention deletes the mind's finished work, its history and the owner gate's record after 90 days while open work
stays, so a longer window is cut to the last 90 days and the report says so; resolved concerns go after 30 days
and the learning log keeps its newest entries, and a window reaching past what they keep says their counts leave
the rest out. The report opens with three lines (what helped, what is unproven, what looked harmful), then a paragraph on the window,
whether the mind is on and at which level, which faculties are off and which sections show a problem, then one
section per faculty:

- **On or off**, as the running code has it under the instance's `colonymind.yaml` (the file's settings; the
  shell's environment never overrides them). A faculty whose setting is on but that cannot act is reported off,
  with the reason and what switches it on: the mind switched off (semantic recall, the memory blocks and the
  status block run without it, and the learning rules in part: only those for remembered statements), goals with
  the drives off, skills with lessons off, wishes in the model's own words with wishes off, and the feelings while
  the feeling rules stand in for them.
- **What it did in the window**, in plain sentences, from the stores' own records (the table below).
- **Cost** where it is recorded: the mind's own model calls and tokens (the `cost_tokens` its rows carry and the
  calls a night records). A task's work in the body is counted by the body, not here.
- **Latency** where it is recorded: tasks from forming to an outcome, messages to the owner from forming to
  sending, the owner's answer to an ask, the owner gate's waits, a night's run.
- **Looks wrong**: errors (failed tasks, messages that could not be sent, a night that hit an error or did not
  finish, a tripped breaker, failing health checks), stalls (a task with the body for over a day since it was
  handed over, with no outcome; an ask past its expiry still open), a faculty that is on but left no trace, one
  that is off yet shows activity in the window (formed before the switch changed, or a sidecar that has not
  reloaded), and semantic recall on with no embedder (neither an embedding endpoint nor the local embedder,
  `router.embed_provider: local`, whose model the section names).
- **Benchmark**, with `--bench DIR`: the latest paired evidence for the faculty on each family in DIR, read as
  below, or "not measured yet".
- **The setting** that switches it off (or on when it is off); for initiative, the mind's own off switch.

| Section | Setting | What the report reads |
|---|---|---|
| Initiative | `mind.enabled` | intentions formed (tasks, messages, goals), asks granted, declined, expired or waiting, task outcomes, messages sent and to whom by role, notices, the owner's own ratings, breaker and health notices, stalls |
| Drives | `mind.faculties.drives` | intentions by drive (context: they form with it off too), the weights in use (off, every drive weighs the same), a drive set to zero |
| Deliberation | `mind.faculties.deliberation` | model calls on open-ended concerns: those that became work and those that formed nothing |
| Goals | `mind.faculties.goals` | goals adopted and how they ended, steps worked |
| Broadcast | `mind.faculties.broadcast` | open concerns now and, as the mind picks them, the top three with thought left that widen recall and those the Mind section shows (none with it off); concerns the drives raised and resolved, as context (they never make it look active) |
| People | `mind.faculties.people` | messages to contacts (check-ins, messages the owner asked it to pass on), replies, questions about who is who, composition cost |
| Owner outreach | `mind.faculties.outreach` | findings, open loops, care and answers sent to the owner, the owner's reactions, follow-ups, a pause |
| Owner gate | `mind.faculties.owner_gate` | messages reviewed, each once by its last verdict: let through, made to wait, held for the digest (and released since), dropped; how many waited first; second opinions (whether a message arrived is the sending path's record) |
| Semantic recall | `mind.faculties.semantic_recall` | no record here; whether an embedding endpoint is set |
| Earlier statements by meaning | `mind.faculties.priors_by_meaning` | no record here; whether an embedding endpoint is set (it needs semantic recall on) |
| Memory blocks | mandatory context fitting | no record here |
| Status block | `context.status` | no record here |
| Nightly consolidation | `mind.faculties.consolidation` | nights run and finished, contradictions, per-contact digests, episode summaries, the questions it asked the owner and how they ended |
| Contradictions by meaning | `mind.faculties.contradictions_by_meaning` | nights that compared statements under different names by meaning and how many names, people left for a later night, a night the embedding model failed, an embedding model never measured on the labelled pairs (it needs the nightly consolidation and semantic recall on) |
| Self-narrative | `mind.faculties.self_narrative` | nights it was rewritten, the account's size and date |
| Opinions | `mind.faculties.opinions` | opinions formed, revised and withdrawn (and which the owner asked for), views held: the owner's views the running mind lists, so one whose premise was superseded or whose source is gone is not held |
| Lessons | `mind.faculties.lessons` | lessons learned, activated, replaced, retired and kept; lessons served in the owner's turns |
| Learning rules | `mind.faculties.learning` | the learning log: restatements turned away or counted as witnesses, preference changes held for or decided by the owner, items cooled or reinforced |
| Skills | `mind.faculties.skills` | skills kept, loads, skills written or removed |
| Wishes | `mind.faculties.wishes` | wishes made and what became of them |
| Wishes in its own words | `mind.faculties.wishes_named` | wishes that came with the model's own words: shown beside the typed capability, or withheld |
| Feelings | `mind.faculties.affect` | the feelings the mind holds now, topics it is frustrated about |
| Feeling rules | `mind.faculties.affect_rules` | no record here |
| Identity check | `mind.faculties.drift_check` | checks run, steady or moved, the latest result |

**Read-only.** The report may run beside a running sidecar. It never opens a store through its own class (those
create tables and migrate). Each SQLite file is opened read-only with `query_only`: a WAL database a process has
open (its `-shm` index is present) through SQLite's shared-memory protocol, so the report reads a consistent
snapshot while the sidecar writes, and a WAL database nobody has open as an immutable file, since a read-only
connection would otherwise create the `-wal` and `-shm` files beside it. A missing store is said and never stops
the report. The only file it writes is `--out` (`P/reports/instance.py`).

### What the benchmark says

```sh
colonymind report bench DIR [--format md|html] [--out PATH]
```

DIR is a results directory: the plan directories `colonymind models paired plan` makes, with the `report-NNN.json`
a run writes, and any `*.report.json` copies of `colonymind models paired report --json` beside them. Each plan is
read once, in its latest view (the newest file, then the highest report number). For every contrast the report
says what was compared, how many scenarios, how often each arm passed, the difference and its 95% interval, how
sure that is (the pre-registered test's p, or the smallest difference that many scenarios would reliably show)
and what it means for the faculty.

A contrast is read as the one faculty its two arms differ in, from what each frozen arm profile switches (read
with the worker's own `mind_section`, `context_section` and `arm_environment`, so a declared profile or an overlay
counts too), and is oriented so the difference is the faculty's: with it minus without it. A profile names only
what an arm switches, and the rest is the shipped defaults of the image that ran it. Five switches have shipped
both ways (off before faculties shipped on unless shown harmful: the owner gate, the learning rules, wishes, the
identity check and the status block), so where one arm sets one of them and the other leaves it to the defaults,
whether the two differ depends on the image. A report that records the defaults its image shipped
(`shipped_defaults`) is read with them; one that does not is read both ways, and a contrast that reads differently
is not read as any one faculty's ("a difference the report cannot place"). An arm's name never stands in for that
record: `full-plus-owner_gate` run today runs what `full` runs. The plan's pre-registered rule is read in the
faculty's direction (`P/reports/bench.py`):

- **helped**: the exact sign test and the interval clear it in its favour, as the rule demands of a treatment (p
  below alpha, at least the minimum number of winning scenarios, the interval above zero);
- **looked harmful**: the same rule clears it the other way (proven), or it did worse by more than the plan's
  non-inferiority margin (10 points; a point estimate, reported as "points to harm", not proven);
- **unproven**: anything else; by the policy an unproven faculty stays on;
- **not measured**: no usable contrast (a contrast with an unattributed scenario is unavailable, and says so).

Across families a faculty's newest reading on each family counts: harmful on any family wins, then helped, then
unproven. A family is its exact dataset version and set (`mind-memory-1` and the harder `mind-memory-2` are two,
and a development set and a held-out set of one family are two), so a newer reading on one never hides an older
one on another. ColonyMind against stock Hermes, one drive's diagnostic (its weight at zero), the same arm twice (the
noise floor) and a contrast of more than one change are shown and never read as one faculty's. The report opens
with the same three lines as the live report.

### The faculty suite

```sh
colonymind models paired suite \
  --dataset-dir /private/families/memory-dev --dataset-dir /private/families/outreach-dev \
  --output /private/results/faculties-01 \
  --native-config /private/candidate-hermes.yaml --native-binding candidate \
  --comparison-policy /private/paired-policy.json \
  --container-image registry.example/agent-benchmark@sha256:IMAGE_DIGEST \
  --streams 2
colonymind report bench /private/results/faculties-01
```

The suite measures every faculty on the chosen families in one command. The reference arm is `all-on` (every
faculty on; [the arms](PAIRED-AGENT-BENCHMARK.md#arms-and-profiles)) and the treatment arms are each faculty's
`all-on-<faculty>` ablation (for the feeling rules, which replace the running feelings, `all-on-plus-affect_rules`:
the rules in their place), plus `all-on-initiative` (all-on with the mind switched off) for initiative, whose
measure is the mind itself. `--faculties skills,status` measures only those; `--dataset-version` names an
installed dataset instead of a generated family directory. A plan takes at most eight arms, so each family's
arms are spread over the fewest balanced plans of at most seven, each with the reference: four plans a family for
all twenty-two.
Each plan is the ordinary `colonymind models paired plan` and `run`, run as a subprocess and logged in
`<output>/<label>.log`; the suite records itself in `<output>/suite.json` (`P/qualification/paired_suite.py`).

**Within a limit of concurrent benchmark streams.** One plan is one stream: its episodes run one at a time, as
`models paired run` always runs them. `--streams N` runs at most N plans at once, each plan's steps in order in
its stream (default 1). Choose N from what the model endpoint and the Docker host take without distorting
timings (a shared endpoint is allowed and is recorded in the comparison policy). The suite takes about
`ceil(plans / N)` plan durations, a plan's duration being its arms times its scenarios times one episode: two
families at `--streams 2` are eight plans in four rounds of two. `--dry-run` prints every plan and its commands
and runs nothing. The suite is resumable: rerun the same command, and a plan already made is not made again, a
started run resumes its untouched episodes (`--resume`, never replaying a started attempt) and a plan whose
every arm-episode has started and that has a report is skipped. A run can stop early and still write its report
(a container whose state could not be cleaned up ends it); such a plan is reported as not finished, with the
number of arm-episodes not started, now and on every rerun. A plan that fails is reported and the others go on.
A plan's label comes from its family and its position, not its arms, so a rerun with other `--faculties`,
another dataset or other settings names the same plan directories. Before a plan already there is skipped or
resumed, the suite checks it against the one asked for (its arms, reference arm, dataset path and content,
native binding, repetitions, temperature, seeds, embedding endpoint, container image and comparison policy);
when any plan differs, it names what differs and runs nothing (exit 2, `--dry-run` too). Measure other
faculties in another `--output` or under another `--label`.

## Configuration

```yaml
context:
  status: true                      # the status block in the owner's turn (The status block, above)
mind:
  enabled: true                     # the off switch
  autonomy: suggest                 # off | suggest | standard | trusted
  deny: {tools: [], text: [], commands: []}
  worker_toolsets: [web, file, session_search, memory, todo]
  grants: []                        # standing grants (Standing grants, above); none: every floor effect asks
  actions: {}                       # declared actions (Declared actions, above): {name: {description, class,
                                    #   command, args, target?, amount?, currency?, timeout?}}; a problem is shown
  floor:
    executors: [terminal, code_execution, coding, debugging, browser, computer_use, desktop_ui, delegation]
    tools: {}                       # classes for tools the table does not know: {name: {class: none or a floor class,
                                    #   amount, currency, default_currency, target}}; a problem is shown in state
    money_hosts: [api.stripe.com, api.paypal.com, ...]   # a request reaching one is money
    bulk_recipients: 3              # a fourth distinct person in 24 h from one task or concern asks
    permit_hours: 24                # how long the owner's yes to a held effect stays usable, once
  # One worker run per task: task_types per task type, task_max_runtime_s / task_max_retries for
  # any other type and for a field a type leaves out.
  budgets: {tasks_per_hour: 4, concurrent_tasks: 2, owner_messages_per_day: 3,
            outreach_per_day: 3,    # unprompted outreach to the owner, counted apart from reminders
            contact_messages_per_day: 5, per_contact_cooldown_hours: 24,
            llm_tokens_per_day: 200000, open_goals: 2, goal_tasks: 4,
            goal_horizon_days: 7, task_max_runtime_s: 600, task_max_retries: 1,
            task_types: {research: {max_runtime_s: 1800, max_retries: 2},
                         question: {max_runtime_s: 1800, max_retries: 2},
                         mastery_investigation: {max_runtime_s: 1800, max_retries: 2},
                         goal_step: {max_runtime_s: 1800, max_retries: 2},
                         outreach_followup: {max_runtime_s: 1800, max_retries: 2}}}
  # extra_body on every request a mind task's worker makes; null leaves a field to the provider
  worker_request: {max_tokens: 8192, top_p: 0.95}
  # The model a mind task's worker runs on: the name of one providers (or custom_providers) entry of the main
  # Hermes config, with no fallbacks. null follows the main model and its fallbacks.
  worker_model: null
  quiet_hours: "22:00-07:00"        # owner notices wait; the digest and tasks do not
  ask_expires_hours: 72             # an unanswered ask lapses: nothing is done, and it is no verdict (Asks)
  # How asks reach the owner (Asks): ask messages in any 24 hours before the rest go in one list, when the one
  # reminder goes if no digest comes first, and the pause after the agent's last message (0: none). A value
  # outside its range (1 to 50, 1 to 168, 0 to 240) keeps its default.
  asks: {messages_per_day: 3, remind_hours: 24, quiet_minutes: 10}
  breaker: {failures: 3, window_hours: 24, demotion_hours: 72}
  act_threshold: 0.6                # the ranker's effective-score floor
  digest_hour: 8                    # local hour after which the daily digest goes out
  heads_up_grace_minutes: 30        # after a heads-up went out, the overdue reminder for that row waits this long
  drives: {duty: 1.0, social: 0.5, curiosity: 0.5, mastery: 1.0, upkeep: 1.0}   # 0 turns a drive off and cancels its waiting work
  faculties:                        # one binary flag each; each flag is one benchmark arm
    initiative: true
    drives: true                    # weights, satiation and goal adoption; off = flat priority
    deliberation: true              # the one tool-less call per tick; off = templates only
    goals: true                     # agent-owned goals
    affect: true                    # the agent's own feelings (Feelings, above)
    affect_rules: false             # the affect mechanism arm: every consumer reads its stateless rule
    broadcast: true                 # the top-3 concerns in turn context and recall
    semantic_recall: true           # embeddings in recall when router.embed_url is set; off = lexical only
    consolidation: true             # the nightly consolidation (docs/CONSOLIDATION.md)
    self_narrative: true            # the self-narrative in the owner's prompt and colonymind_self state
    opinions: true                  # the opinion pass, approach views in task bodies, the stance section
    lessons: true                   # lessons: the night's lesson stage, lesson lines, the lesson section, the reflector
    skills: false                   # proven lessons as SKILL.md in ColonyMind's skills.external_dirs entry
    outreach: true                  # owner outreach: the social drive toward the owner (Outreach, above)
    owner_gate: true                # one gate every unprompted owner message passes (docs/OWNER-GATE.md)
    learning: true                  # the learning rules over lessons and claims (Learning rules, above)
    wishes: true                    # the mastery drive's wishes (Wishes, above)
    wishes_named: true              # the model's own words beside a wish's typed capability: labelled, checked, never the grant
    drift_check: true               # the identity battery and drift detection (Identity check, above)
    # Read when set, on otherwise (a new colonymind.yaml does not name them): the embedding model beyond recall.
    contradictions_by_meaning: true # the night also compares statements under two names close in meaning
    priors_by_meaning: true         # the claim extractor is also offered earlier records found by meaning
  owner_gate: {topic_window_hours: 12, repeat_window_hours: 24, conversation_minutes: 10, spacing_minutes: 10,
               max_wait_minutes: 60, confidence: 0.75, second_opinion: true, judge_timeout_s: 30}
```

`identity.yaml` holds the constitution (`agent.name`, `agent.values`,
`agent.boundaries`) and may list `agent.interests`; each interest becomes a
seeded interest at startup. `faculties.semantic_recall` is a real switch:
with it off the embedder stays off (`COLONYMIND_EMBED_PROVIDER=skip`) even
when `router.embed_url` is set. Learning writes `mind.db`, the feedback
multipliers and the intention rows; nothing the mind learns writes
`colonymind.yaml` or `identity.yaml`.

## The CLI

```
colonymind mind status              enabled, level, queues, breaker, the affect line
colonymind mind log [--limit N]     the audit log, newest first
colonymind mind why <id>            drive, evidence, decision, Hermes ref, outcome, verification
colonymind mind asks                open asks with their codes (their ids for the console and the API)
colonymind mind asks-report [--since 7d] [--json]
                                    how the asks went over a window (Asks: The asks report)
colonymind mind show <code>         send an open ask to the owner's chat again, where their answer counts
colonymind mind yes <code>          approves nothing: a yes is the owner's, in their own chat (see show)
colonymind mind no <code>           decline an ask (recorded as the operator's, never the owner's answer)
colonymind mind rate <id> <verdict> actioned | dismissed | ignored | useful | not_useful | wrong
colonymind mind level [<level>]     show or set the autonomy level
colonymind mind reset <class>       reset a tripped breaker
colonymind mind off | on            the off switch (off works with the sidecar down)
colonymind mind tick                run one tick now
colonymind mind stats               the in-vivo panel over the audit log
colonymind mind concerns            what is on the mind: drive levels, open concerns, the broadcast set
colonymind mind goals               the agent-owned goals that are open
colonymind mind wishes              the mastery drive's wishes: evidence, the owner's word, measured outcome
colonymind mind interest <topic>    seed an interest for the curiosity drive
colonymind mind outreach [on|off]   owner outreach: its state, a pause until turned on, or on again
colonymind mind consolidate         run the nightly consolidation now (docs/CONSOLIDATION.md)
colonymind mind narrative           the self-narrative as the owner's prompt section renders it
colonymind mind opinions [list|show <id>|withdraw <id>|reconsider <id>] [--query Q] [--history] [--reason R]
colonymind mind lessons [list|show <id>|retire <id>] [--all] [--reason R]
colonymind mind grants [list|add <id>|revoke <id>] [--class C] [--scope S] [--tools T] [--match TARGET]
                       [--max-amount N] [--currency C] [--per-day N] [--per-month-amount N] [--expires DATE]
                                    add adds nothing: it prints the mind.grants entry (Standing grants)
colonymind mind allows [revoke <id>]
                                    the owner's standing allows; revoke brings the asking back (Standing allows)
colonymind mind drift [--run]       the identity check: the last run, the trend, the owner agreement; --run runs one
colonymind report faculties [--since 7d] [--format md|html] [--out PATH] [--bench DIR]
                                    what each faculty did over a window, read-only (Reports, above)
colonymind report bench DIR [--format md|html] [--out PATH]
                                    the paired benchmark's results in plain words (Reports, above)
```

## The API (`/v1/mind`, one bearer key)

| Route | Body | Returns |
|---|---|---|
| `GET /dispatch` | | a JSON list of approved task intentions: `id`, `kind`, `type`, `drive`, `dedup_key` and `idempotency_key` (`mind:<id>`), `title`, `body`, `assignee` (`colonymind-act`), `recipient`, `reason`, `max_runtime_seconds`, `max_retries`, `goal_mode`, `goal_max_turns`, `parent_goal_id`, `created_at`, `expires_at` |
| `POST /dispatch/{id}/bound` | `{hermes_ref, hermes_kind?, status?, bound_at?}` (idempotent: the same ref may arrive again) | the audit entry |
| `POST /outcome` | `{id, hermes_ref, hermes_kind?, status, outcome?, final?, summary?, error?, verified?, run?, block_kind?, consecutive_failures?, completed_at?, observed_at?}`: `status` is the kanban status, `outcome` (`done`, `blocked`, `failed`, `cancelled`, `uncertain`) the body's reading of it; `final: false` (a requeued failed run) is logged, not settled | the audit entry; 404 for an unknown intention |
| `GET /outbox` | | a JSON list of messages: `id`, `kind` (`notice` for the owner, `message` otherwise), `type`, `dedup_key`, `recipient`, `recipient_is_owner`, `recipient_handles` (`{gateway, address, is_primary, verified}`, read afresh at every pull), `text`, `title`, `created_at`, `expires_at` |
| `POST /outbox/{id}/sending` | `{target?, at?}` | the audit entry; 409 unless the message was ready and the claim names a target (`no_target`: the body resolved no handle, the message stays ready and is offered again; a claim nobody settles is `uncertain` after 10 minutes) |
| `POST /outbox/{id}/sent` | `{result: sent | failed | uncertain, error?, hermes_ref?, summary?, at?}` | the audit entry |
| `POST /observations` | the body's board: `{observed_at, board, body, counts, stale_tasks, blocked_tasks, goals, mind_tasks}` with `idle_s` per task (docs/HERMES-ADAPTER.md), or the flat `{observations: [{kind, id, title, assignee, status, age_hours}]}`; stale owner tasks and goals are duty inputs | `{accepted, kinds}` |
| `POST /guard` | `{tool, args, session | session_id, run, task_id, recipients?, cwd?, effect?, context?, ...}`: a messaging tool's recipient is read from `args` (`contact_id`, `platform` + `target|chat_id|to`, or stock `target="platform:chat_id[:thread_id]"`); `recipients` are the contact ids an effect reaches later (a delivering cron job), each authorized with `may_contact` and the message budgets; `cwd` is the directory a mind run's command runs in (part of the call a permit binds to), `effect` the plugin's reading of the call (the sidecar classifies again) and `context` what only the plugin can read (a vault item's kind, the id of the cron job a removal or update names) | `{allow, action: allow | block | ask, reason, effect?, grant?, permit?, held?}`: a held effect answers `block` with `held` (the ask's code) |
| `GET /grants`, `POST /grants`, `POST /grants/{id}/revoke` | `{grant, reason, by, contact_id?, session_id?, message?}` to add, from the owner's conversation (it becomes a `grant_confirm` ask; `by` only labels), `{by?, contact_id?}` to revoke | every grant with source, limits, uses and validity; `{ok, status: asked, code, id, text}` or `{ok, status: revoked, text}`; 403 `approve_required` for an add that names no contact, session or message, whatever the credential; 403 not the owner, 422 a malformed grant, 404 an unknown grant |
| `GET /allows` | | `{allows, resetting, applies, text}`: each standing allow in force with the words the owner agreed to, the ask it came from, when and its uses; `resetting` the revoked ones whose contact permission reset still waits; `applies` whether the level lets allows apply; the owner's view (a guest gets none) (Standing grants: Standing allows) |
| `POST /allows/{id}/revoke` | `{by?}` | revokes the allow at once (a narrowing, so any credential may): the asking comes back; 404 `unknown_allow` |
| `GET /cron-origin?job_id=&created_at=` | | `{origin: mind \| owner \| unknown, mind, task_id, intention_id}`: whose cron job this is |
| `POST /cron-origin` | `{job_id, task_id, tool_call_id, action, ok}` | the plugin's report once a mind run's cron create or update returned |
| `POST /lineage` | `{child, parent}` | a kanban task a mind run made: it counts with its maker |
| `POST /answers` | `{platform, sender_id?, contact_id?, chat_id, chat_type, turn_id, message_id, arrived_at, text, reply_to_message_id?, reaction?, reaction_removed?, forwarded?, voice?, own_account?, ledger?}`: the owner's own message as the host saw it before its model ran; `ledger` is what reached that chat before it (`[{at, outbox_id?, message_id?, kind?, ordered?}]`, at most 200; none: only a reaction or a quoted reply binds) (Asks) | `{read, key, path, reading, asks, settled: [{id, code, sentence, status}], approved, later, shown, clarify, undone, refused, unbound, acknowledged, context}`, or `{read: false, reason}`; 403 `owner_not_granted` from a platform the credential is not granted the owner on, 403 `platform_not_allowed` from `companion` for any credential but the instance key. The words are never stored |
| `POST /asks/{id}/shown` | `{platform, chat_id, message_id, at?, asks?}`: a host put these open asks in front of the owner in one message (several: a list, in that order) | `{ok, asks, message_id}`: that message now binds in that chat: on one ask, a reaction or a quoted reply answers it; on several (a list), a quoted reply binds as that list for 24 hours and a reaction on it is ignored; 422 without the platform or the chat, 404 an ask that is not open, 403 `platform_not_allowed` for `companion` from any credential but the instance key |
| `POST /asks/{id}/show` | | the ask, by its id or code, sent to the owner's chat again as its own message: `{ok, id, message, text}`; 404 not an open ask, 409 no owner to show it to |
| `GET /permit-asks` | `state` (`open`, `settled` or `all`), `limit` (50; at most 200) | `{asks: [view], state}`: the hands permit asks with their immutable ids, full terms, offered choices and, once settled, the winner (channel, choice, time); only an owner channel relay's `answer` credential, never the instance key (403 `answer_credential_required`; [hands-permit-answer-contract.md](hands-permit-answer-contract.md)) |
| `GET /permit-asks/{id}` | | one hands permit ask's view, whatever its state; 404 `unknown_ask` |
| `POST /permit-asks/{id}/answer` | `{choice, platform: companion, contact_id?, sender_id?}` (one of the two) and an `Idempotency-Key` header, values exactly as spelled: the owner's explicit answer from an owner channel, through its relay's own `answer` credential | `{result: settled, ask_id, status, outcome, winner, settlement, replayed}`, settled through the same guarded settle path as `/answers` (the first commit wins), and the owner told on RCS (`ask_answered`); only an `answer` credential; 409 `already_answered` (the recorded winner), `ask_closed`, `idempotency_conflict` or `full_text_not_delivered`; 403 `answer_credential_required`, `platform_not_allowed` or `not_owner`; 404 `unknown_ask`; 422 `invalid_choice` or `invalid_platform` |
| `GET /asks/stats` | `since_hours` (168; at most a year) | the asks report (Asks) and its `text`; the owner's view only |
| `POST /tripwire` | `{kind: answer_forgery_attempt, tool, named: [api.key, answer_route], run, session_id?, refused?}`: the plugin's guard saw a tool call read the instance key or name an answer route; `refused` when it refused the call (a non-owner session or a mind run), else the call ran (the owner's own lane) | `{ok, id, noticed}`: a note in the audit log that says which; the owner is told at most once in 24 hours |
| `GET /log` (`limit`, `status`, `kind`, `since_hours`, `recipient`; `split=true`: `{actions, notes, text}`, the agent's actions apart from its notes and notices), `GET /why/{id}` (404 `unknown_intention`: "no intention `<id>` exists in the audit log"), `GET /log/{id}`, `GET /asks`, `GET /state` (`/status`; with `faculties`, `drives`, `concerns`, `goals`, `interests`, `deliberation`, `affect`, `outreach {enabled, paused_until, per_day, sent_24h, muted, care}`, `lessons {enabled, active, candidate}`, `skills {enabled, generation, owned}`, `grants`, `grants_changed`, `floor_tool_problems` and `actions`), `GET /stats` (with `lessons` and `lesson_use_rate`, the wins over verified uses, and `skills`) | | |
| `GET /wishes?viewer=` | | `{enabled, wishes, stats, text}`: every wish, newest first, with its class, capability, why, cited failures and lessons, state, ask code while it waits, goal, baseline and after windows, outcome and developer request; a guest viewer gets nothing |
| `GET /lessons?status=&uses=&viewer=` | | `{enabled, lessons, uses, skills, text}`: each lesson with its verified tally, with `uses=true` every use in the 90-day window, and the skills ColonyMind keeps with their loads; a guest viewer gets nothing |
| `POST /lessons/{id}/retire` | `{reason, by?}` | the retired lesson; 404 unknown, 409 already closed |
| `POST /skills/used` | `{skill, session_id?, task_id?}` (the plugin's `on_skill_lifecycle` forwarding) | `{ok, counted, loads}`; only `colonymind-*` skills are counted |
| `GET /narrative` | | the self-narrative `{enabled, text, sections: {interests, strengths, recent, stances}, cites, updated_at}`; `enabled: false` and empty text until the mind keeps one or while `faculties.self_narrative` is off |
| `POST /consolidate` | | runs the nightly consolidation now and returns the night's record (`colonymind mind consolidate`); 501 `consolidation_not_available` on a sidecar without it |
| `GET /drift` | | the identity check `{enabled, runs, last, pending, owner_word, trend, agreement}` (Identity check, above) |
| `POST /drift/run` | | one identity check now, audited like any run: `{id, trigger, verdict, moved, score, battery, answers, signals, substrate, owner, ...}` |
| `GET /concerns`, `GET /goals` | | the workspace (open concerns, the broadcast set, drive levels) and the open goals |
| `POST /interests` | `{topic, why?}` | a seeded interest the curiosity drive researches |
| `POST /owner-gate` | `{topic, text, urgency: low \| normal \| high \| critical, source?, id?}` | the owner gate's verdict on an outside alert `{decision: send \| wait \| digest \| drop, reason, stage, until, scores, facts, id, gate}`, recorded; `{decision: send, gate: off}` with `faculties.owner_gate` off (docs/OWNER-GATE.md) |
| `GET /owner-gate` | `limit`, `since_hours` | the gate's audit `{enabled, settings, counts {decision, stage}, decisions}`, each decision with its inputs and scores |
| `POST /outreach` | `{state: on \| off}` | owner outreach's state `{enabled, paused_until, per_day, sent_24h, muted, care}` after the switch; off pauses unprompted outreach until turned on; `{enabled: false}` with the faculty off; 422 for another state |
| `POST /asks/{code}/yes`, `POST /asks/{code}/no` | none (a body is ignored) | yes: 403 `approve_required` for every credential (Asks: console and API answers); no: the audit entry, recorded `by: operator` |
| `POST /off {reason?}`, `POST /on`, `POST /tick`, `POST /rate {id, verdict}`, `POST /level {autonomy}`, `POST /reset {cls}` | | |
| `GET /opinions?q=&contact_id=&by=&history=&limit=`, `GET /opinions/{id}`, `POST /opinions/{id}/withdraw`, `POST /opinions/{id}/reconsider` | `{reason, contact_id?, by?, correction_id?}` for the two controls | `{enabled, opinions}`, `{opinion, history}`, `{revision_id, status}`; audience-filtered, owner-only controls (docs/OPINIONS.md) |

`GET /dispatch` and `GET /outbox` also record the body's last pull; when it is
older than five minutes the tick stops forming intentions until the body is
back (`POST /tick` forces one).

Test seams: `COLONYMIND_MIND_CLOCK_OFFSET_SECONDS` shifts the mind's clock
(the 72 h ask expiry is checked by restarting the sidecar three days ahead,
`scripts/ci_mind_loop.sh`), and the paired benchmark passes its own shifted
clock in (`colonymind.qualification.native_memory_worker.serve_mind`), builds
its Mind with the sidecar's own factory (`colonymind.mind.factory.build_mind`:
the same `CommitmentExtractor`, so a forced tick drains capture first, and the
same people, appraisal and router wiring) and mounts the same mind routes
(`mind_routers()`), and writes an `identity.yaml` whose owner handle is
the capture platform's home channel, so an owner reminder is a delivery the
grader attributes to the mind rather than a board task. The drives family
runs the `full`, `full-drives`, `full-broadcast` and per-drive arms
(docs/PAIRED-AGENT-BENCHMARK.md).
