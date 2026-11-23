# Nightly consolidation

Sleep-time compute for the mind (architecture 3.1, 4.1, 4.2, 4.8; build plans M8 and M9). Once per
night crossed, the sidecar consolidates what the time since the last run left in its stores: the
self-narrative delta, the lessons verified results taught, contradictions settled (the newer statement kept),
per-contact digests, the agent's pages and episode summaries. It runs inside the token budget, writes only to stores
that already exist, and its whole record is one audit row.

Code: `sidecar/colonymind/mind/consolidate.py` (`Consolidation`); the schedule is in
`sidecar/colonymind/mind/tick.py` (`Mind._schedule_consolidation`).

## When it runs

| Condition | Rule |
|---|---|
| The boundary | The nightly boundary is a local time of day: the start of `mind.quiet_hours` when they are set, else 03:00 local; `COLONYMIND_AGENT_TIMEZONE` decides "local" |
| Due | The boundary fell in `(last run, now]`. `mind_state["consolidation.last"]` (no half-life) holds the moment of the last run in `updated_at` and its local date in `text`; a store without it is marked at the mind's first start, so a fresh store never consolidates before its first night has passed, whatever hour it started at. So: once per night crossed, never twice for the same night, and once for a machine that slept through the boundary |
| Switches | `mind.faculties.consolidation` or `mind.faculties.lessons` (the night runs when either is on, or `mind.faculties.wishes` or `mind.faculties.work_records`, and each stage checks its own flag: the lesson stage `lessons`, the wish stage `wishes`, the work stage `work_records`, every other stage `consolidation`, so `full-consolidation` keeps its lesson stage and `full-lessons` its consolidation), `mind.enabled` / `colonymind mind off` (the tick returns before the schedule), a router that supports function routing, and `Authority.tokens_allowed()` (the shared day budget) |
| How | The 60 s timer tick starts one `asyncio` task off the tick's lock, before the body-stale check (the body, Hermes, is not needed), and goes on. A forced tick (`POST /v1/mind/tick`: `colonymind mind tick`, the plugin's `tick()`, the benchmark body) awaits a night it found due, at most 300 s (`CONSOLIDATION_WAIT_S`), so what the night wrote is there when the tick returns; a night still running then keeps running in the background. `Mind.stop()` and `colonymind mind off` cancel a night in flight at once |
| Forcing | `colonymind mind consolidate` (`POST /v1/mind/consolidate`) runs it inline now, whether or not a night was crossed; never with the mind off or with all of `faculties.consolidation`, `faculties.lessons`, `faculties.wishes` and `faculties.work_records` false (`{skipped: off}` / `{skipped: consolidation and lessons off}`). It too marks the moment of the last run |

`Mind.state()["consolidation"]` reports `{last, running, last_tokens}` (the tokens of the last
finished run, whose row the marker names). Every run is its own audit row, written `done` when the
run starts (result: "did not finish") and charged after every call, then given its summary at the
end; no row is ever left running. A night cut short (a restart, `mind off`) keeps what it wrote and
leaves the scheduling marker unmoved, so the night is still due and runs again at the next due tick.
Each stage's entry marker is committed before it runs, so the next run starts with the stages
least recently entered; each stage retains its own idempotence checks.

The forced tick reports the night as `incomplete` when it ended with errors, rather than
reporting `done`. Failures increment `mind_incomplete.<task>` in the owner view of
`GET /v1/mind/status`; the audit retains the stage counts and errors.

## What runs, in order

After settling claims, every enabled stage runs unless the budget or whole-night deadline stops
the night first. Stages run oldest first by the night in which they were last entered; never-entered
stages go first. Stages entered in the same night use the order below. Entry is recorded before
the stage runs, even if it fails or is cancelled. Thus stages not reached after a cut go first next
night, and a night that enters every stage restores the plain order for the following night.

0. **Settle the claims** (no call of the night's own). Statements still waiting for claim
   extraction, such as one said minutes before the boundary, are extracted first, so the night
   reads them tonight rather than a day later: a claimable job is processed with the mind's router,
   one the projection worker holds is waited for, all within 120 s (`CLAIM_SETTLE_S`). This is the
   claim projection's own work, not the night's, so it is not charged to the night's share; a job
   that fails goes back to its retry time and is not waited for; with `COLONYMIND_SOURCE_CLAIMS`
   off (no claim extraction on this sidecar) the step does nothing. The night's record counts
   `claims_settled`. A forced tick's wait (300 s) covers it, so the paired body's night tick sees a
   statement from the turn just before it whatever the model's latency.

1. **Narrative delta** (`mind.faculties.self_narrative`; one call). The computed sections are
   rebuilt from the stores and stored: `self.interests` (the three strongest seeded and declared
   interests, each citing its own record, `interest:<slug>`), `self.strengths` (the two task
   types with the most work over 30 days: `research: 3 done, 1 failed, 2 verified of 4`, or
   `weak at <type>` when failures outnumber successes, citing up to two intention ids; only the
   agent's own actions count, never an intention it dropped or deferred) and
   `self.stances` (three of the judgments store's current revisions, each citing
   `judgment:<revision id>`, until M7's opinion store replaces the reader). Only `self.recent`
   ("the last 7 days") is model-written, by delta edits of at most 4 lines. Its evidence is the
   agent's own actions of the last 7 days (`audit.is_action`: a task, goal or message it decided
   to act on or ask about; never an internal note or a notice) and their recorded findings, all
   under the intention's own id, so every `recent` citation is an id `colonymind_self why`
   explains. Actions addressed to anyone but the owner and contradiction questions (they quote
   what people said) are left out; the plugin renders the narrative only in the owner's own
   sessions. Every line must cite ids from that evidence that also exist; a line that cites
   anything else is dropped and counted in `rejected_lines`. A stored line lasts only while every
   action it cites was formed in the last 7 days: an older one disappears at the next render, with
   or without a night, so `recent` never presents stale activity. When the narrative is rendered,
   each `recent` line is followed by where its cited actions stand now (their outcome, or `queued`,
   `in progress`, `awaiting the owner`, `deferred` while they have none), so a line that claims more
   than happened is read beside the log's own word. Lines end with their citations in square
   brackets. A citation is one of five kinds: a
   plain intention id (the agent's own action), or a record reference `interest:<slug>`,
   `judgment:<revision id>`, `turn:<turn_id>` or `claim:<id>`; nothing else resolves.
2. **Lessons** (`mind.faculties.lessons`; one call, `P/mind/lessons.py`). The packet: the owner's
   own sessions with a turn since the last review (`mind_state["lessons.scanned"]`; a first night
   reads the last 7 days) and at least two owner messages, at most 6 sessions of their last 8
   turns, the owner's messages labelled `t1..tn` and the agent's replies beside them, each session
   with the lessons used in it; the agent's tasks and goals of the last 14 days that an owner
   verdict, an external check or a Hermes failure with a reason stands behind and that the night
   has not read in that state (`result_metadata.lessons_seen`), at most 6, labelled `i1..in`; and
   the current lessons they bear on, at most 8. With nothing to read there is no call. The call
   (`mind_lessons`, at most 1,200 output tokens) returns its verdicts (the owner messages that
   judge the agent's earlier work, right or wrong, quoted) and `add`, `supersede` or `retire`
   operations, each validated before anything is written (docs/MIND.md, Lessons): a verdict only
   for an owner message that follows an agent reply in its session, citations inside the packet, an
   owner citation only of a reported verdict and with the owner's exact words quoted, a strategy
   only from an owner verdict or an external check, a pitfall also from a Hermes failure, a
   retirement only on a verdict or a verified failure, never from an unverified result or a
   contact's session. What passes is admitted `active`, with the strongest cited verifier, the
   corrected value split into `retrieval` or `knowledge`; each checked verdict scores the lesson
   use of the owner message just before it in its session; what was read is marked, so the next
   night starts after it. The stage ends with the lesson review (retire under a 0.4 win rate after
   five verified uses, activate a candidate after a verified win) and the skills sync. Counts:
   `lessons_admitted`, `lessons_superseded`, `lessons_retired`, `lessons_activated`,
   `lesson_uses_scored`, `lesson_ops_rejected`, `skills_written`, `skills_removed`. Every step is
   idempotent, so a night cut short admits each lesson once when it runs again.

In the plain order, **work** follows lessons (`mind.faculties.work_records`): up to 16 schema-validated
3-8-word titles in one budgeted call, plus exact links from the lessons stage's
vetted verdict inbox. Pending titles and verdicts have durable retry watermarks;
a failed title call leaves rows untitled. See [the work contract](WORK.md#nightly-titles-and-verdicts)
for identity checks, erasure fencing and the reported six-session/eight-turn
verdict coverage limits.

After work in the plain order, **the wish** (`mind.faculties.wishes`, on by default; at most one call,
`P/mind/wishes.py`): the mastery drive may name one capability the agent lacks for one failure class
with at least two verified failures in 14 days, citing them. A valid wish is a concern the tick forms
into one owner ask; the night records a refusal in `mind_state["wishes.refused"]` (docs/MIND.md, Wishes).

3. **Contradictions** (one typed decision per pair). The rule recall already applies: two live scalar
   claims about the same `(subject, predicate)` with different values and overlapping validity
   (quoted preferences and derived claims never count). The same fact filed under two keys is found
   too ("I work at Acme" as `works at`, "My employer is Globex" as `employer`): each night every
   person's live claim keys (`key_text`: the subject, "I" for the speaker, and the predicate) are
   embedded in one batch per person, and two of that person's claims under different keys whose
   embeddings are at least the bound alike (cosine), with different values and overlapping
   validity, are a pair as well. Claims of two people are never compared, and an
   episode's key (a record label) is never embedded. The vectors exist only inside that call: no
   cache and no store, so nothing derived from a claim outlives the night and erasure has nothing
   more to reach. Without the sidecar's embedder, or when it fails, takes longer than 30 seconds for
   one person or answers the wrong number of vectors, the night pairs exact keys only, as before, and
   so it does with `mind.faculties.contradictions_by_meaning: false` (the pairs by meaning ship on).
   The whole pass is bounded too (`KEY_EMBED_PASS_S`, two minutes, and never more than a quarter of
   what is left of the night's 15 minutes), so a slow embedder cannot starve the relation decisions,
   digests and episodes: the persons with the newest statements are embedded first, the pairs found
   before the time runs out stand, and the rest wait for a later night (`key_embedding_deferred`).
   A bound belongs to the embedding it was measured on, since one model's cosines say nothing about
   another's. Each is chosen on a labelled dev set of 60 synthetic pairs, half one fact that holds
   one value at a time and half two facts that can both hold
   (`colonymind/mind/contradiction_keys_dev.json`, pinned by its hash):
   `python -m colonymind.mind.contradiction_keys` scores every pair with the night's own key text,
   keeps the lowest threshold whose candidates are at least 80% same-fact pairs and that finds the
   most, and reports the default bound's precision and recall beside it. It embeds as the sidecar
   does: every option (`--provider`, `--base-url`, `--model`, `--dims`, `--request-dims`) defaults
   to the sidecar's own `COLONYMIND_EMBED_*` setting, so a deployment that asks its endpoint for
   fewer dimensions is measured at that width, and the report records the embedding identity it
   measured (the fingerprint the vector index keeps). `--record FILE` adds the measurement to a
   bounds file; the shipped one is `colonymind/mind/contradiction_keys_bounds.json`, whose entries
   keep every pair's score so each choice can be recomputed. The night uses the bound measured for
   its own embedding (the same model at the same width, on the pinned dev set). An embedding whose
   measurement kept no bound at 80% same-fact pairs finds no pairs by meaning
   (`key_bound_below_floor`); one never measured uses `KEY_SIMILARITY_BOUND` (0.87), no looser than
   any bound measured, and the night counts `key_bound_unmeasured`. A deployment can keep
   measurements of its own, for an embedder the release did not measure or at a precision floor it
   chose (`--precision-floor`, recorded in each entry): `--record` into a file of its own and name
   that file in `COLONYMIND_CONTRADICTION_KEY_BOUNDS` (the `environment:` section of
   `colonymind.yaml`). For the embeddings it measured, the night reads that file's bound instead of
   the shipped one; a name that is not a file leaves the shipped bounds in force, with one warning.
   Before anything is raised, each such pair (every newer statement against every older one it
   overlaps, newest first, at most 8 a night: compatibility is not transitive; the exact-key pairs
   and the pairs by meaning are two lists, each newest first, taken in turn and never reranked) is
   put to one typed
   decision on the night's budget (`mind_claim_relation`): does the newer statement replace the
   older (`replaces`: they cannot both hold now) or add to it (`adds`: a detail, a refinement, a
   narrower case, another item of a list)? An addition is recorded as compatible (`mind_state`
   `claims.relation:<pair>`, and the learning log, stage `contradiction_check`) and never asked; an
   answer outside the two leaves the pair for a later night. The read is Instinct's `claim_relation`
   point ([DECISIONS.md](DECISIONS.md)): this call is its existing path, and a model tier may only ever
   record `adds`. Only a pair decided `replaces` is a
   contradiction, and the night settles it itself: it keeps the newer statement and asks the owner
   nothing. The older statement and every live copy of its value (the same value for the same
   period, under that value's own key) are retracted by the night's note (`retracted_by` is
   `note:<id>`; nothing is deleted), each in the learning log (action `declined`, stage
   `replaced_by_newer`), and the night counts the pair in `contradictions`. A pair filed under two
   keys is found again from its recorded decision, so a night without the embedder settles it too
   and never judges it again. A question an earlier build raised
   (`dedup_key = reach_out:contradiction:<id>`) is withdrawn: its concern resolves (the tick
   resolves one that had formed no message yet, so it never forms), one not yet
   sent (deferred, asked or queued) is cancelled by the mind (`verified: check`, no verdict), and
   one already sent is closed by the question code at the next tick (docs/MIND.md, Questions).
   Every plain summary the owner reads (`mind_plain_summary`: a digest item, an ask notice) is
   written in the language and script of the item it shortens and never translates it; names,
   identifiers and quoted words stay verbatim. Its no-cut check reads the item's own script: a
   sentence ends at that script's full stop, question or exclamation mark (。？！ among them), and
   in a script written without spaces between words (Chinese, Japanese, Thai) each letter counts
   as a word.
4. **Per-contact digests** (`mind.faculties.people`; at most 6 calls). A digest has one home, the
   contact's own record (`digest`, `digest_sources`, `digest_writer`, `digest_withheld_at`), and one
   writer path, the contact store's `set_digest(contact_id, text, sources, writer=...)` (awaited when
   it is a coroutine). A contact store without `set_digest` (the one before the people milestone)
   gets no digest, and nothing is stored anywhere else. Candidates are only contacts still marked stale, whatever their last interaction;
   the nightly job never scans recent contacts for a blanket rebuild. The background worker normally
   repairs these contacts first. A stale template may also be rebuilt by the night. The model sees that person's own live claims (at most
   40, with their ids) and the previous digest and returns at most 600 characters of what the
   person has told the agent, plus the claim ids it used; a digest citing none of the claims it was
   shown is rejected (and tried again the next night). The digest is written in the language and
   script of the person's own statements (when they mix languages, the one most of their statements
   use) and never translates them; names, identifiers and quoted words stay verbatim, and a previous
   digest in another language is written fresh from the statements instead of carried over. The
   rule is in the prompt: no code decides which language a text is in. The stored `digest_sources` are the live
   claims the digest was built from, whichever the model cited, so an unchanged contact costs no
   call and its digest does not drift. If one of them stops being live while the model writes, the
   digest is not stored (`digests_stale`).
   The mind renders no section of its own: the people milestone's `colonymind-person` section
   renders the contact's digest in that contact's turns.

   **Lineage and erasure.** Both writers record what a digest rests on: the tick's template digest
   cites the claims whose lines it renders (`digest_writer: template`), the night's cites the claims
   it was built from (`night`). Before the `colonymind-person` section, the owner's
   `colonymind_people inspect` or a host contact route (`GET /v1/host/contacts`, `/contacts/{id}`,
   `/contacts/resolve`, and the writes that answer with the contact) returns a digest, one query by
   id checks that every cited claim is still live: present, not superseded, not expired, not retracted, not from a source whose
   projections were erased, and still that person's own (from a person-scoped source of theirs that
   no attribution correction moved or invalidated, the rule both writers read claims by; a moved
   source read again under its new person gives its claims their old ids, and those are the new
   person's words). A digest that cites a claim that is not
   live, or a source that is not a claim, is withheld (a contact route then returns neither `digest`
   nor `digest_sources`, and returns none either while the lineage cannot be read): its text leaves
   the contact's record and `digest_withheld_at` marks it for a rebuild, so a claim that comes back
   later (a restore) never revives the old text. Source forgetting now appends a curated tombstone
   and uses the shared recall fence; it retains historical digests and performs no store cleanup.
   An attribution correction runs the lineage check on both people's digests when it moves their sources,
   before anything reads them. A rebuild never reads a withheld digest, or one citing a claim that
   went, as its previous digest, just as with a cooled claim. Fact insertion, replacement, withdrawal and deletion queue a durable per-contact invalidation
   in the source ledger, committed with the claim change. The background worker clears that contact's
   old digest and renders a template from its current claims after 30 seconds without a fact change.
   Both template and generated digests use this refresh path, including when the mind is off.
   At most 60 refresh attempts start in a rolling hour; this budget survives restarts and counts failures.
   Readers withhold invalid lineage immediately. Background lineage checks also detect expiration
   and attribution changes. Pending invalidations survive a failed contact-store write, and a fact
   change during rendering leaves another rebuild pending. Unchanged contacts retain their summaries.
   `GET /v1/host/health` counts the withheld digests (`recall.digests_withheld`). A template digest
   stored without its claims (sources the marker `["template"]`) cannot be checked: the store
   migration withholds it, and the next tick renders it again with its claims, the mind on or off.

After the digests in the plain order, **the pages** (at most 7 calls, `P/pages/night.py`): the agent's page
on itself and its page on each person ([pages](SOCIAL-STATE.md#pages)) are written again. The stage first
removes every page line whose source turn was erased or forgotten (`page_lines_removed`). A page is due
when its person has a turn newer than the page's last rewrite: the agent's own page first, then at most
six people, the page written longest ago first; the people not reached wait for a later night. One call
per page: the model sees the page as it stands and the newest conversation (at most 200 turns within
6,000 characters, each turn under a short id) and returns the lines as they should stand now, at most
twelve. A line that cites no turn it was shown is dropped and counted in `rejected_lines`. A person's
page is written from that person's conversation, in the language and script of their own statements (the
digest's rule); the agent's own page from what the agent itself said, to anyone. The agent's boundaries
with a person and the lines it wrote itself are kept as they are. A call with no answer leaves its page
as it is, due again the next night. With `mind.faculties.people` off only the agent's own page is
written. The night's record counts `pages`, and each run of the stage writes one `check` line to the
recorder (source `night.pages`, detail `pages-<n>`; `failed` when the stage raised).

5. **Episode summaries** (at most 8 calls). Sessions with at least 3 person-scoped turns since the
   last finished episodes stage (at least the last 24 hours, at most 7 days), oldest first by the first
   turn in that window newer than their newest surviving summary (or their first turn in the window
   if they have no surviving summary). Ties use contact id, then session id. A session needs a
   summary only if its newest turn is newer than its newest surviving summary,
   or it has no surviving summary. The newest surviving summary is the one whose row id (below) sorts
   last as text among the session's surviving summaries: the latest date in the id, not the time the
   row was stored; a summary row under an id of another form is not seen. After the local date has
   gone below a date already used, a summary written under the earlier date does not become the newest,
   so a session whose newest turn is newer than its summary of the later date can be summarised again
   on each night below that date while its turns are still in the window (whose anchor stays where it
   is while the stage is unfinished), from the same turns if no new one came; on the night of that date
   it waits, a summary with a later date ends the repeats, and nothing is lost: no stored summary is
   replaced.
   Empty transcripts use no
   call and leave no backlog. A session with
   a surviving summary under tonight's date waits until the next night for another summary.
   The window's anchor advances only when no candidate is left untaken and no call failed; otherwise
   unfinished work stays within the bounded window, with untaken candidates reported as
   `episodes_pending`. When more sessions keep arriving than the stage takes in a night, for longer
   than the window, they are still taken oldest waiting first, so a session that talks every day can
   stay behind older ones until a quieter night, and each such night's record shows the backlog as
   `episodes_pending` in its `counts` and `episodes: backlog` in its `errors`. The first anchor comes
   from `consolidation.last`, independently of stage entry
   order. Each session is summarised in at most 120 words, in
   the language and script of the person's own messages in that session (the same rule as the
   digest, never a translation), and written by `Autobiography.record(contact_id=...)` as the ledger row
   `mind:episode:<session>:<date>:episode_summary`, or `mind:episode:<session>:<date>:<digest>:episode_summary`
   when the first id is forgotten (`<digest>`: the first 16 hex characters of the SHA-256 of the list of
   turn ids the summary is made from), using the writing night's local date, under **that contact** (`session_id="mind"`,
   `scope="person"`, `derive_claims=False`): the person's later sessions recall it, and it never
   becomes a claim. The summary carries the session's turns as its source lineage (the ledger's
   supplied sources), so when the person has any of those turns forgotten, the ledger's erasure
   closure erases the summary with it; a later night may summarise what is left.

Every input query excludes the mind's own rows (`SELF_TURN_SQL`: `session_id<>'mind' AND turn_id
NOT LIKE 'mind:%'`). The same exclusion applies to the commitment extractor's "Recent conversation":
the autobiography is the agent's record, not the person's conversation.

Claim dedupe happens where the night reads claims, never in the store: of the scalar claims
repeating one `(subject, predicate, value)` over one period (the same validity and event time),
the contradiction and digest stages see only the newest witness, the rule recall's
`current_group` already applies (`distinct_values`); quoted preferences are never folded. The
commitment extractor's prior claims use the same rule (`source_projection.one_witness_per_value`):
the same value over two periods stays two claims there, so a correction can name either. Because
the night pairs only the newest witness, keeping the newer statement retracts every live copy of
the older value (`source_projection.witness_key`: the same value for the same period), so no older
copy stands in its place.

## Budget

A model answer cut off at its length limit gets one retry on the same binding at its `maxTokens`, only
when the night's remaining budget covers it. Both attempts' reported usage is charged. A retry that
cannot finish leaves that work incomplete; it does not silently count as a completed stage.

| Stage | Calls per night | Answer cap |
|---|---:|---:|
| Narrative delta | 1 | 600 |
| Lessons | 1 | 1,200 |
| Contradictions | <= 8 decisions, and a summary only for a statement too long to quote | 40 per decision |
| Per-contact digests | <= 6 | 300 |
| Pages | <= 7: the agent's own page and at most 6 people | 600 |
| Episode summaries | <= 8 | 250 |

The night may spend at most `mind.budgets.learn_share x mind.budgets.llm_tokens_per_day` tokens
(0.25 x 200,000 = 50,000 by default). The share is the learning work's: the lesson call is one of
the night's calls, charged to the same row. Before every call the run checks that what is left of that
share still covers the call (the larger of what it may generate and the biggest call so far) and that
`Authority.tokens_allowed()` holds. What a call may generate is its answer cap, or on a binding that
reasons before answering the larger completion cap the router sends it: the answer cap plus that
binding's reasoning allowance (`reasoningTokens`, [function routing](FUNCTION-ROUTING.md#reasoning-allowance)); the call's real usage, as the endpoints reported it, is written at once to
`cost_tokens` on the night's `note/consolidation` audit row: an answered call's (`response.usage`, with any
attempt that failed before it) and a failed call's too (a completion that ran out while thinking still used its
whole cap, and the router's error carries what was reported), so the shared day budget and
`colonymind mind stats` (`mind_tokens`) see it. Each call has the router's function deadline
(clamped to 10 minutes) and the whole run is bounded by 15 minutes; a stage that fails is recorded
in the row's `context.errors` and the next stage still runs. Calls are labelled by `context.task`
(`mind_consolidate_narrative`, `mind_lessons`, `mind_claim_relation`, `mind_plain_summary`, `mind_consolidate_digest`,
`mind_consolidate_page`, `mind_consolidate_episode`) and carry
`workload: background`, which the router keeps in its call record (`routing_status`
`recent_calls`) and never sends. The overhead row's background tokens come from the audit rows'
`cost_tokens` (`colonymind mind stats`: `mind_tokens`): the paired harness observes requests on the
wire, where the label is not.

## What it writes

| Store | Key or row |
|---|---|
| `mind.db` `mind_state` | `consolidation.last` (`updated_at` = the moment of the last run, text = its local date); `self.interests`, `self.strengths`, `self.recent`, `self.stances` (text = the section, causes = up to 5 cited ids) |
| `mind.db` `mind_state` | `consolidation.stage.<name>` (`updated_at` = start time of the last night that entered the stage); `consolidation.episodes.last` (`updated_at` = last finished episodes window, initially taken from `consolidation.last`); no cycle record |
| contact store | `digest`, `digest_sources`, `digest_writer` of each digested contact, through `set_digest` (only a store that has it); a withheld digest's text, through `withhold_digest` |
| `mind.db` `concerns` | none new; a `question` concern of an earlier build (`dedup_key reach_out:contradiction:*`) is resolved |
| ledger `source_claims` | `retracted_by = note:<id>` on the older statement of a replaced pair and on every live copy of its value |
| `initiatives` (the audit log) | one `note/consolidation` row per run (`cost_tokens`, `context` = the run's record: counts, calls, tokens, rejected lines, errors) |
| ledger `turn_sources` | `mind:episode:<session>:<date>:episode_summary`, or `mind:episode:<session>:<date>:<digest>:episode_summary` when that id is forgotten, under the episode's contact; `mind:lesson:<id>:<event>` under the owner (session `mind`, `scope='session'`) |
| ledger `page_lines`, `page_rewrites` | the night's lines of each page written again (`writer` = `night`), each with its source turns, and the moment of each page's last rewrite ([pages](SOCIAL-STATE.md#pages)) |
| `<instance>/recall-questions.jsonl` | nothing is added; a question whose source turn was erased or forgotten is removed ([the instance's own question set](RECALL-HYBRID.md#the-instances-own-question-set)) |
| `mind.db` `mind_state` | `lessons.scanned` (the newest owner turn the lesson stage read); `skills.generation` |
| `initiatives` | `lesson_use` notes' scored results (`result_metadata.use`), `lessons_seen` on the rows read, `lesson_activated` / `lesson_retired` notes |
| `<instance>/skills` | with `faculties.skills` on, the promoted lessons' `SKILL.md` and ColonyMind's manifest |

Nothing here writes `colonymind.yaml` or `identity.yaml`.

## Reading it

- `GET /v1/mind/narrative` and `colonymind mind narrative`: `{enabled, text, sections: {interests,
  strengths, recent, stances}, cites, updated_at}`. `text` is at most 800 characters (3 interests,
  2 strengths, 4 recent lines, 3 stances), one line per entry prefixed `interest:` / `strength:` /
  `recent:` / `stance:`, each ending with its ids in brackets. It rides in every model request of
  an owner session, so it is sized by the overhead budget
  (`tests/hermes_adapter/test_overhead_budget.py`). With `self_narrative` off: `enabled: false` and an empty text. With `consolidation` off
  and `self_narrative` on the computed sections still render (the `full-consolidation` arm keeps a
  narrative; `full-self_narrative` has none).
- `POST /v1/mind/consolidate` and `colonymind mind consolidate`: the night's record (`local_date`,
  `calls`, `tokens`, `budget`, `counts`, `done`, `errors`, `rejected_lines`, `id` of the audit row),
  or `{skipped: running}` while a scheduled night is in flight.
- `colonymind mind why <id>` on the night's row, and `colonymind mind log --kind note`.
