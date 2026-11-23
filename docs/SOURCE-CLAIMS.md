# Source-grounded temporal assertions

Ordinary attributed turns now retain a small factual assertion projection in the canonical source ledger. Recall can distinguish an explicit correction, a dated change and an unresolved disagreement. This is an incremental memory behavior, not a truth engine or completion of all temporal reasoning.

Machine-authored reports and other retained text quotations can receive an
[attributed source annotation](SOURCE-ANNOTATIONS.md) without manufacturing a
USER assertion or rewriting the original evidence.

## Ingestion and provenance

`TurnIdempotencyLedger.record_source` stores the source and queues its projection in the same transaction. The new `source_claim_jobs` and `source_claims` tables use the existing `turn-idempotency.db`. There is no additional database or vector index. Existing sources are not automatically backfilled; `colonymind memory reextract` forms them again on the owner's command ([Re-extraction](#re-extraction)). Historical checkpoints remain quotations because their individual messages do not attest the speaker and occurrence time.

The background consumer uses the named extraction role and its configured local fallbacks. The role's `timeoutSeconds` bounds each candidate and `deadlineSeconds` bounds the entire routing attempt. The consumer captures that total deadline and allows five additional seconds for dispatch overhead; a longer explicit background budget is no longer clipped by a fixed 40-second wrapper. Public routing defaults remain unchanged. Older tier-only configurations use SMALL with a 20-second timeout and no escalation. Concurrency is one. It starts with the sidecar; `COLONYMIND_SOURCE_CLAIMS=off` stops extraction while retaining sources and previously derived records. An unavailable local role leaves durable pending work. Before each message, the worker renews its existing owned lease for the captured request bound plus 30 seconds to commit. This accommodates multi-message sources and role reloads without a heartbeat process. Process loss retries incomplete work; stale consumers cannot renew, commit or finish a reclaimed job.

Extraction accepts at most six assertions per string user message of at most 12,000 characters. Each must retain an exact contiguous source quotation, its message hash and character span. Subject and value must occur in that quotation; the one exception is a subject named in an earlier message of the same conversation (extraction version 19, below), which grounds only who or what the subject is. Unsupported, uncertain or rejected extraction remains raw source evidence. No confidence number is promoted to truth. The result records the extractor model alias and extraction version; changing the model does not change stored assertions.

For typed messages of at most 500 characters, extraction version 15 offers an
`evidence_refs` entry for the complete current message. The extractor can return
its `evidence_ref` instead of copying the quotation. The consumer resolves that
reference to the current source bytes before applying the existing subject,
value, date, correction and admission checks. Unknown references and conflicting
literal quotations are rejected; literal output remains supported without fuzzy
quote repair. The reference grants no truth or authority and does not bypass
preference review or change the existing whole-source episode policy. Longer
messages continue selecting bounded literal passages. Audio transcripts retain
their existing segment quotation and provenance checks and receive no such ref.

Version 2 also requires a useful memory category and an inspectable reason for
later recall. These are unverified model judgments. Explicit personal-disavowal
phrases are checked against the full message so an extractor cannot clip them
away and turn a self-example into a preference. This is a limited English
validation rule, not general entailment checking. Persistent consumers reject
reasoning-only and unfinished provider output. See [memory quality](MEMORY-QUALITY.md).

The ordinary-turn outbox captures `occurred_at` once when a new turn enters durable delivery, unless a lifecycle timestamp was supplied. Retries reuse that value. The fallback timestamp means turn capture, not independently verified speech or event time. Source ingestion time remains separate. An explicit event date is separately stored as `event_at`. Unknown historical occurrence times remain unknown.

## Corrections, changes and conflicts

Assertion keys are scoped to the attributed contact and normalized subject/property. A first-person assertion means that contact. Normalization preserves Unicode and compares whole normalized values; substring containment is not agreement.

| Input | Stored effect | Recall effect |
| --- | --- | --- |
| Independent assertions give different values | Both keep their source and recording time | Overlapping validity becomes an unresolved conflict |
| Explicit correction names an existing assertion | New record links to the old ID; old record is retracted | Old value cannot become the current answer |
| Explicit change has an effective time | Old interval closes; new interval starts; supersession links remain | Historical queries can recover the earlier value |
| A change happens during a queried calendar day | Both intersecting intervals remain | Temporal history, without labelling disjoint intervals a contradiction |
| Source is erased during or after extraction | Claims tied to removed message hashes are deleted | Late extraction cannot restore them |
| A correction is erased | The older retraction link remains without erased text | Erasure does not silently revive the old value |

A newer timestamp alone never supersedes a conflicting assertion. Generated property names and prior-record matching can still fail; an unmatched correction becomes an independent assertion. The model is not permitted to invent a predecessor ID. The implemented projection does not grant authority, alter relationships or execute tools.

Extraction version 18 (`source-claims-v18`) changes three things: the prior records,
the key list and vague event dates. Existing claims keep the version that formed
them; only messages extracted from then on use it. The extractor can link a correction or change
only to a prior record it is offered, at most 16 per message. They come from two lanes. The keyword lane
offers the live claims of the turns a keyword search of the new message finds,
or, when it finds none, the person's most relevant recent claims. For a person
with many claims nearly every message finds something, so a paraphrased change
("We relocated to Munich" after "My office is in Berlin") was never offered its
record. When the sidecar has source vectors, a vector lane runs the existing
source vector search with the new message, in the person's scope alone (a
session's checkpoint notes carry no claims and are left out before ranking, so they
cannot take the lane's places), and offers the live claims of the turns it finds, in
its order. The lanes are taken in turn, one
record each, keyword first, each claim and value once: rank order, not a fused
score. The vector lane makes no vector of its own, so nothing new needs erasing;
without vectors, or when the search fails, the keyword lane is offered exactly as
before, and so it is with the lane switched off (`mind.faculties.priors_by_meaning:
false`; it ships on, and the claim projection reads it as
`COLONYMIND_PRIORS_BY_MEANING`). Commit still validates every `prior_claim_id` and review still checks the
link. A worker that runs away from the loop owning the vector store (the paired
benchmark's claim worker thread) has no vector lane.

The extraction call also lists the person's recent distinct claim keys
(`known_keys`: subject and predicate, newest first, at most 20), so an assertion
about a property the person already has reuses its key instead of naming it
again. The list holds names only, read through the same scoped reader as the prior
records: never another person's key, an erased or reattributed message's, a
retracted or superseded claim's, or an episode's content-derived identity. A key
supplies no value and is not a record to correct; the review does not see the list.

Extraction version 19 (`source-claims-v19`) adds subjects named in an earlier
message. "My sister moved last month." followed by "She lives in Denver now."
formed no usable claim, because a claim's subject had to occur in its own
quotation. The extraction call is now also offered the conversation's previous
user messages (`earlier_messages`): the typed messages of the same person, session
and person scope said before the current one, at most 4, said within the 6 hours
before it, each whole up to 1,000 characters and a longer one by its end (all three
are builder's choices, not measured bounds). Never another person's or
conversation's message, an assistant's, a transcript's, or an erased, reattributed
or annotated one. What the assistant said after an offered message, before the
user's next one, comes with it as context only (`assistant_after`: the last 4 of
its messages, each by its end), so the calls can tell whom the current message
answers ("Where does your mother live?" between the two above makes "She" the
mother): no id names those words, so no subject is ever taken from them, and after
a message that is not offered they are not shown. When the current quotation names
its subject only by a pronoun or leaves it implied, a claim may copy the subject
from one of them and name the
exact words (`subject_basis`: the message's id and the quote); the extractor is told
to abstain when more than one subject could be meant. The words must be an exact
span of that message (case, spelling and extent as written) and must be the
subject's own; the claim keeps the span of the subject's words
(`subject_basis_message`: the message, its hash, the span and which occurrence of
those words in the message it is), marked as subject identity only. Sealing a
secret in that message later, or releasing one, rewrites it in place and moves the
words after the value; readers then find the same occurrence of the same words, so
the claim, a claim a correction retired and one re-extraction set aside keep their
basis. The value, dates, operation and evidence still come from the current
quotation, as for every claim. The review is shown the earlier messages (with what
the assistant said after them) beside such a proposal and told to reject it when
another subject could be meant, the current message answers the assistant about
someone else, the reference is to someone else, or the value belongs to another
subject; the commit checks the
span again, and that the message is the same person's and conversation's, said
before the current message and within the window. A correction or change of such a
claim in a later conversation keeps its basis (the same message), as a correction
keeps an inherited subject; the speaker ("I") is never named by an earlier message.
With no earlier message offered, the request is exactly the one version 18 sent.

Recall shows the basis beside the claim (`subject_basis`, with the words and their
source) and cites that message as a source of the assertion, but ranks the claim by
its own quotation alone: the candidate format and the calibration fingerprint are
unchanged. The prior records' keyword and vector lanes also offer the claims whose
subject a message they found named, so a later "My sister moved to Austin" is
offered the claim the pronoun formed. Erasing or reattributing the earlier message
removes every claim it named, with what depends on them; annotating it makes them
ineligible, as for a subject inherited from a prior claim. A partial correction
that names the subject itself but carries such a claim's words ("Correction: My
sister bought a blue one." after "She bought a red Ford.") depends on that message
too: the carried words are about the subject it named, so the correction and the
views formed on it go, or become ineligible, with it, and recall cites the message
beside the words (the contribution's `subject_basis`). The same holds for words
carried from a claim whose subject a prior claim named. Such a claim never folds
into another as a witness, but with the learning rules a correction of its value
retires it with that value's other live rows.

`colonymind report claims [--since 7d] [--json]` says how this works on an instance, in
counts only (read-only; no words, names or ids): the claims formed from the messages
captured in the window, by extraction version, so a build's counts stand beside the
one before it; how many took their subject from an earlier message, and what became
of each (it stands, it is hidden because the message that named its subject was
annotated since, a correction retracted it, or a later change replaced it); and the
proposals the extraction refused for the earlier message they named, by reason
(`subject_basis_not_exact`, `subject_basis_mismatch` and the rest). A claim erased
with its message, or with the message that named its subject, is gone and not
counted. A correction of such a claim can mean it named the wrong subject.

A partial correction can carry unchanged parts of the exact prior value. For
example, changing a glove quantity does not require repeating an unchanged brush
quantity. Each changed portion must occur in the current quotation; carried
portions retain their original quoted sources. The existing semantic review
checks the complete update against both records before the prior claim is
retired. This does not permit unsupported new values or merge unrelated facts.

Recall includes these supporting quotations, source versions, report times and
timezones. A word such as "tomorrow" keeps its original source clock when a later
correction changes only the hour. This provenance does not itself resolve a
compound deadline. Further corrections preserve each contribution's original
source. Erasure or changed attribution removes dependent projections, while
other raw correction messages remain; an annotation makes disputed support
ineligible. None of these operations revives a retracted value.

## Recall and time

Canonical source hits locate assertion keys. Recall expands each key into its scoped, time-appropriate evidence before the existing shared reranker and context budget. An unresolved conflict is one atomic candidate: the packet cannot retain just the winning side or truncate away its qualifications. Raw source chunks have the projected spans removed; each remaining excerpt is still a contiguous quotation, including when source chunks overlap. Graph summaries do not participate in automatic context or explicit canonical memory search.

A claim's key can locate its evidence too. A claim was reachable only when the
words or the meaning of the message that states it matched a question; a
derived full-text index now also holds each claim's subject, predicate (its
underscores read as spaces) and value, written as the stemmed keyword lane
writes messages (English stemming, CJK runs as their two-character pieces),
one row for each place the claim is stated: its anchor and each witness
message of a folded claim. The `claim_key` recall lane
([hybrid recall, Keyword lanes](RECALL-HYBRID.md#keyword-lanes)) gives a match
as a source hit: the words of the message the claim quotes, read back from the
canonical source and checked against the claim's span. Recall expands that hit
like any other, so what is shown is the grounded assertion bundle, never the
key. "Who is my employer" reaches `employer: Acme` from "I just started at
Acme", whose words it does not share. It helps only a question that uses the
key's words: "where do I work" shares none with `employer`, and only the
vector lane can reach it. A wrong extraction now also gives wrong hits; the
reranker and its cutoff still judge them.

The key rows follow the claims in SQL, whoever changes them. Triggers delete a
claim's rows when the claim or one of its witnesses is deleted or moved (an
erased message, an erased correction and every claim built on an erased one, a
re-anchored or unfolded claim), and a deleted source takes its rows. Rows are
written only from the claims as they are: in the claim commit's own
transaction, after an erasure, when the secrets migration or the release of a
sealed value has finished rewriting (a changed message moves every claim it
states), and when the ledger opens. A rejected claim is
never stored and has no rows, and a dry run leaves none. A retracted or
superseded claim keeps its rows: its key still locates the property, and
expansion picks the value current at the question's time. An episode has no
rows (its subject and predicate are fixed words and its value is the reported
message itself, which the keyword lanes already search). The first start after
the upgrade builds the index from every claim and logs how many claims it
indexed and how long it took; claims an older version wrote after a rollback
are indexed on the next start. `TurnIdempotencyLedger.rebuild_claim_keys()`
rebuilds it from the claims, with the same rows each time.

The injected assertion cards share repeated exact quotations from the same
canonical message version. A packet-local `evidence_ref` links each assertion to
its quotation, source, message hash, speaker and report/recording time. Assertion
IDs, values, event precision, validity intervals, correction ancestry and opening
anchors remain attached to their cards. Different message versions are never
merged. Internal assertion JSON is rendered once as data; raw source text stays
quoted, including text that resembles JSON. The shared character budget measures
the complete rendered packet, so a repeated passage does not consume the budget
again for each property. This changes presentation, not retrieval ranking or
whether a claim is true. A report timestamp does not establish event order.

For a current-state query, a short message of at most 2,000 characters remains
one quoted evidence unit when partial extraction would separate its clauses.
This reuses the complete-procedure source check: every stored assertion must
still be eligible, without a superseded/retracted sibling or conflicting peer.
The quotation carries its canonical message hash and property-history opening
anchors. Its unclaimed clauses are no longer independently ranked against the
asserted clause from the same message. Changed messages retain the existing span
suppression; historical queries retain assertion/time semantics. If the complete
message cannot fit the injection budget, an opening notice replaces it instead
of presenting a convenient prefix. This preserves context but does not establish
that a model will interpret the quotation correctly. Non-procedure audio
assertions retain their structured evidence cards and exact segment/recognizer
lineage; a machine transcript's display prefix does not trigger short-message
expansion. A complete audio procedure keeps its full quoted conditions and
attaches the represented claims' exact segment bases as metadata, still labelled
derived and unverified. An oversized procedure supplies only an opening notice,
with no partial procedure or segment-basis payload.

Valid time, event time and recording time are different fields. Interval comparisons use canonical UTC timestamps. Supported query dates are ISO dates/datetimes, full English month dates and anchored today/yesterday/tomorrow; event queries also support trailing hours/days. Calendar days use the resolved contact/communication timezone. A historical assertion without a known validity start is not certified for that date. Raw quotations remain labelled with unknown validity. An unprojected source captured inside an event window is labelled `source_occurrence_only`, which does not establish the event's time.

Relative dates resolve against the turn's own time (or an explicit search's `as_of`) as the span they name, in local calendar units of the contact's timezone, end exclusive:

- "last week" and "this week": the calendar week (Monday to Monday) before this one, or this one; "the last week", "the past month" and "the past year": the trailing week, month or year up to now;
- "last month", "this month", "last year", "this year": that calendar month or year;
- "N days ago": that day; "N weeks ago": the calendar week that held the day N weeks back; "N months ago", "N years ago": that calendar month or year (N a whole number of one to three digits, or in words up to twelve; "a week ago" is one);
- "the weekend" and "over the weekend": the latest Saturday and Sunday that have begun; "last weekend": the one before them while a weekend is under way, else the same; "this weekend": this week's;
- "in March": the latest March that has begun; "March 2025", "in 2024" and "the summer of 2025": that month, year or season;
- "last summer" (any season): the latest such season that has ended. A season resolves only where the contact's timezone places it in one hemisphere, outside the tropics (southern summer is December to February); otherwise it is not read at all.

"since X" runs from X to now, as an event window. "before X" runs from the earliest time to X and "between X and Y" from X to the end of Y, where X and Y are any of the dates above. A question with a resolved date is a validity window, or an event window when it asks what happened, was seen or recorded, as for explicit dates.

Other ranges ("after", "until", "through"), "next week" and "the previous month", "N units ago" with a count the parser does not read whole ("1.5 months ago", "1,000 months ago", "1000 weeks ago"), several dates in one question, "before" or "between" without dates the parser reads, and an explicit date that does not resolve are labelled unresolved instead of silently selecting the first date: an unresolved question admits no claims. The parser does not understand all natural-language temporal questions and reads English only; an explicit search can name its own window instead (`window` and `as_of`, docs/RECALL-HYBRID.md, "Recall and time"). It does not implement historical transaction-time queries such as reconstructing exactly what the system believed before an ingestion date.

Time constraints come from request text outside valid JSON objects/arrays,
code fences, blockquotes and quoted report passages. A timestamp inside supplied
evidence does not require the source to have been captured at that instant.
A quoted time operand, such as `recorded on "2026-03-12"`, `"last 2 hours"` or
`"last week"`, is read as the same date unquoted.
These syntax rules affect only temporal interpretation; lexical and
semantic retrieval retain the complete original query. Unmarked narrative and
malformed or truncated pasted structures can remain ambiguous.

One key expands to at most eight distinct values. A larger group supplies a compact incomplete-history marker with a source/claim anchor; it does not choose a value. The shared five-result/character budget still applies. An oversized atomic assertion bundle can similarly supply an opening anchor when the marker fits. Repeated identical values do not crowd out a distinct conflict. This remains bounded discovery, not a guarantee that every relevant source is found. Graph records without source lineage are outside canonical recall.

## Complete source opening and event precision

Extraction version 5 retains `event_time`: the exact source expression, known
precision, or explicit unresolved status. A supported optional event phrase such
as "before sending this message" does not erase an otherwise supported assertion.
It bounds the event before source occurrence, without assigning that timestamp
to the event. Calendar-day expressions retain their day interval. Invented date
text and unresolved required `valid_from_text`/`valid_to_text` still reject a
proposal. The existing semantic reviewer checks the complete source and its
conditions; this representation does not independently prove entailment. Earlier
assertions without precision metadata remain labelled legacy precision unknown.

From version 18, a vague event date is kept as the whole period it names, never its first day:
`precision` is `period`, `period` is `year`, `month` or `season`, and `start` and
`end_exclusive` are the period's local month boundaries in the message's time zone,
as UTC instants. "in 2015" is that year and "March 2015" that month; "last year",
"last month", "this year" and "this month" count from the message's own local date,
and "last summer" is the latest summer that had ended when the message was sent.
Seasons are meteorological (three whole months). A season resolves only where the
time zone's location, read from the time zone database's own location table, lies
outside the tropics: `UTC`, `Etc/*` zones and tropical locations leave it
unresolved. A season with a year resolves only when it stays inside that year (a
northern winter spans two). Relative periods carry a `basis` with the rule, the
message time and the time zone. A period is event time only: `valid_from_text` and
`valid_to_text` still need a supported date. Both claim readers accept any resolved
interval, a calendar day or a period: an event-date question matches a claim whose
interval overlaps its window, labelled `query_time_unresolved` when the interval is
wider than the window, and an episode dated to a period outside the window is never
offered as an undated one.

No Hermes tool opens a source since 1.10 (the adapter's source-read tool went with
its other native tools): the model reads the evidence recall puts in its context. A
client with the `read` access opens one through `POST /v1/host/memory/read`, which
requires exact canonical selectors: `person_id`, `source_id`, `source_version`,
`session_id`, and optional `source_view=assertions` plus an anchored `claim_id`. Source pages contain
at most 4,096 characters of serialized canonical messages and applicable
attributed corrections. History pages contain at most eight assertions, including
superseded/retracted status. Histories above 10,000 scoped assertions return an
explicit read-limit error. `next_offset`, `offset_unit` and `read_revision` support
continuation; changed evidence requires restarting at offset zero. A partial page
is not a complete procedure. Graph IDs are not canonical source IDs.

Opening checks current scope, version, attribution and annotations. A client
applies the returned source versions and watermark before it uses a delayed
result: after an erasure or a new correction, it opens the source again rather
than replaying the old result as fresh evidence. Actual reader completeness and
answer quality still require behavioral qualification.

The memory-provider wrapper now distinguishes persistent state from source evidence, preserves uncertainty and permits clarification when a material contradiction remains.

## Operation and qualification

The experimental v11 extractor represents substantive experiences and comparisons
as quoted episodes. The processor selects one exact passage and explains its
future use; it does not synthesize subject/value fields for the episode. The
existing source table retains it with a content-derived record identity. When the
episode quotes the entire eligible text message, deterministic source checks
replace the second model review. The extractor still judges usefulness and
correction references. The distinct `source_admission` marker records
`whole_source_quote_unverified`; it does not invent a review or verify the report.
Selected excerpts from longer messages and segmented audio keep context review. An explicit correction selects an offered prior episode, reuses that identity and
retracts the mistaken report through the existing claim lineage. Its original
source remains an identity dependency, so erasure or attribution changes revoke
dependent interpretations; erasing the correction does not revive the old value.
For the first correction, the original quote is labelled `prior_episode_report`
and read alongside the correction. Corrected or withdrawn details are not current;
unchanged clauses remain attributed earlier context, not independently verified
facts. Withdrawing the whole report withdraws every detail. Original event and
report dates stay attached to that original quotation, never copied into the
correction. The store does not synthesize a merged current report.

For successive partial corrections, the root quote is instead labelled
`episode_history_incomplete`. It omits intermediate corrections and cannot
establish current details. The existing assertion-history reader exposes retained
revisions and their links. If a revision is unavailable or withdrawn, the reader
must preserve that gap rather than reconstruct it from older text. Cumulative
interpretation remains a reader responsibility; this projection does not claim
to materialize or validate it. An unavailable predecessor is labelled
`incomplete_revision_chain` even when pagination has returned every retained row.
When lexical retrieval finds only an earlier report, prior-episode selection
looks up its current retained revision through the existing episode identity.
The original quotation supplies topic context, not a stale correction handle.
Lookup remains within the contact's visible sources and existing candidate
limits; erasing or reattributing a revision cannot revive a retracted report.
The correction wire supplies the offered episode ID and current exact evidence;
the stored predecessor supplies its representation and memory kind. Correcting
one count does not turn the episode into a separate structured fact. Structured
branches cannot select episode IDs, and contradictory explicit types or extra
fact fields are rejected rather than silently reinterpreted.
`Reported episode` is a record label, not a person or world-model entity. The
passage preserves units, conditions, attribution and uncertainty within the
existing 500-character limit. An episode may supply an exact event-date expression
from its quotation. The existing date parser resolves supported expressions using
the source clock; report time remains separate. An unsupported or unquoted optional
date cannot discard an otherwise valid exact report. Unquoted date metadata is
removed and counted in `ignored_episode_date_count`; its invented text is not
stored. Missing dates stay unknown rather than inheriting a date from a different
quotation. Relevant episodes with unknown time remain available to event-date
queries as `query_time_unresolved` evidence; known events outside the requested
interval stay excluded. A calendar-day observation is matched as an interval,
not as an event at midnight. Partial overlap across source and query timezones
also carries `query_time_unresolved`, since an unknown point within the source
day cannot certify that the event happened inside the query's narrower window.
Material that cannot fit with its essential context
stays available as source history.

Structured facts and procedures keep their existing representation and temporal
conflict handling. Episodes are independent reports: sharing a topic does not
automatically overwrite an earlier report or resolve contradictions. A correction
must have an explicit correction cue and select a current, admitted episode
reference. Source identity and predecessor eligibility are rechecked at commit.
Selecting the same reported experience remains an extractor judgment. A different
incident cannot retract the earlier one. The existing explicit source-annotation
path also remains available for owner corrections.
There is no new database, worker or model call. This representation is undergoing
behavioral qualification and does not enable automatic opinions.

`GET /v1/host/memory/sources/claims/status?contact_id=...` uses the existing request scope and reports recent job states, attempts, model/version, errors and claim counts. Unavailable extraction leaves quotations usable; it does not block ordinary turn ingestion or the next conversation.

### Dry run

`POST /v1/host/claims/dry-run` shows what a message would become, with nothing
written. The body names a person and supplies the text: `{person_id, session_id?,
text, occurred_at?, timezone?, prompt_only?}`. `occurred_at` (ISO 8601 with a zone,
default now) is the clock relative dates count from; `timezone` defaults to the
person's. It offers that person's prior records (both lanes) and key list, runs the
extraction call, every validation step and the review, then the commit's own checks
for a transient source in a transaction that is always rolled back, so each verdict
is the one a real commit of the same input would reach. The response lists the
offered priors and keys, each accepted proposal (`new`, with the claims it would
retract or supersede and any preference change it would hold for the owner, or
`restates`, with the live claim it would add a witness to), each rejected proposal
with its stage (`validation`, `review` or `commit`) and reason, the extraction
counts, and `usage`: the calls made and the tokens they reported. It is offered the
earlier messages of the named session said before `occurred_at`, as the worker would
be (`offered_earlier_messages`). No claim, job,
witness, learning-log row, source or vector survives it. With `prompt_only: true` it
returns the two prompts and the response schema extraction would send, with no model
call, even when no model is configured (the full dry run then answers 503).

The text is sealed first, as ingestion seals every message: a detected secret's
reference stands in it before the keyword search, the vector lane's query embedding
and any prompt see it, so no model is sent the value. The references are only
planned under the existing vault key and nothing is stored in the vault. On an
instance that has never sealed a value (no vault yet) a detected value is withheld
instead, since a dry run creates no vault. The text is then checked as ingestion
checks a message: an exact copy of an erased message of that session, in any form it
has had (as it came, sealed, or under a reference its value had before), is refused
with `status: source_erased`, before any model call or prompt. When that check
cannot be made because the vault key cannot be read, the dry run answers 503
(`secrets_vault_unavailable`), as ingestion does.

It reads a person's memory, so it is the owner's: a credential that manages the
instance, never development mode or a credential carried from 1.9.0, and the person
is resolved within the credential's grant (403 otherwise). A full dry run costs the
same model calls as extracting the message.

### Re-extraction

A claim is formed once, when its message arrives, and keeps the extraction version
that formed it. A later extraction version helps only new messages until history is
formed again. `colonymind memory reextract` does that, on the owner's command only (a
change of the extraction role's model alone queues nothing):

```sh
colonymind memory reextract --all --dry-run     # the plan and its estimate; writes nothing
colonymind memory reextract --all --limit 20    # a small run, which also measures the cost of a message
colonymind memory reextract --person <id>       # one person
colonymind memory reextract --status            # the runs and their progress
colonymind memory reextract --cancel            # stop what still waits (everyone's)
```

`--cancel` writes and takes no other option: with `--dry-run`, `--status`, `--person`
or `--limit` it refuses and changes nothing.

It queues the attributed person-scope sources whose claims an older version formed
(their claim job records the version). Imported history and checkpoints have no
claim job and stay quotations; erased sources and sources whose attribution changed
are skipped. The command works on the ledger itself and needs no running sidecar;
the source worker runs the queue on a lane of its own. Before it queues anything it
prints the plan: the people, sources and messages, the claims older versions formed,
and the estimate. The plan names the sources the start queues, chosen the same way
(with `--limit`, the first that no run is working yet); the sources a run already
works are counted apart. The estimate counts the model calls (one extraction a message, and
the reviews at the share measured on finished claim jobs) and, at the rate this
instance measured on its own re-extraction, the tokens and the model time; before any
run, a small run with `--limit` measures that rate. Model time is a floor: the lane
yields to live work. `--dry-run` opens the ledger read-only.

**Order.** One person at a time, and each person's sources in the order they were
said (the occurrence time, else the recording time; then the recording time and the
source id), whatever order the queue holds: a source waits while an earlier source
of the same person waits, so a correction is always read after the message it
corrects. Each message is offered what it was offered the first time: only the
records of sources said before it and of its own source's earlier messages (both
lanes, as a new message is), with the records its own earlier
claims retired offered live again, and a key list led by its own earlier keys, so a
property keeps its name.

**Commit.** The new reading goes through the same extraction, validation, review and
commit as a new message. In one transaction per message its earlier claims leave the
store, the new ones commit and what was decided about the earlier ones carries
forward; any failure (the model, its output, the commit) rolls the message back and
it keeps its earlier claims. The new reading commits whole or not at all: when the
commit rejects one of its claims (a basis that stopped being current while the
message was read), the message keeps its earlier claims (`commit_rejected`) and the
rejection stays in the learning log. A message that keeps its earlier claims, for this
or any other reason the report names, is not marked done and its source is not marked
current: its job ends `kept` (`--status` counts it), the next run queues the source
again, and only that message is read again (a source keeps the messages a run already
read under the current version). An earlier claim that another message also states stays,
anchored at that message. Any other is retired: kept in `source_claim_reextractions`
with the link `reextracted_by` to the claim that holds its place (none when the new
reading has no counterpart). A corrected or superseded claim the new reading has no
counterpart for is not retired: it stays as it was, at its message, so recall still
shows its correction for the words it covers (recall offers words no claim covers as
a plain quotation) and a value read anew from another message is still retired by
it. Erasure reaches retired claims with their message, and
with every retired claim that carries the erased message's subject or words (a partial
correction quotes the words it kept), however many retired or live claims lie between.

**Carry-forward.** The same reading of the same message is the same claim: its id, and
with it every reference to it (an owner's answer, a held change, a decided relation, a
lesson), stays, while its data takes the current version's reading (a vague date
becomes its period). A claim that changed has a successor among the new reading's
claims: the one with the same id, else the same key and value, else the same subject
and value under another predicate, else the same key. The last two are guesses, so
they must be unambiguous: a claim about another subject never matches, and a guess
holds only when it is the earlier claim's one candidate at that level and no other
earlier claim's candidate there. Without a successor nothing is carried to a claim
of the new reading.

- The successor takes the earlier claim's retraction or supersession; a supersession
  it would outlast is reported (`link_contradicted`), not applied.
- A link that pointed at the earlier claim points at its successor. A correction whose
  link the new reading missed still retracts what it retracted, and says so; with no
  successor the link keeps the earlier claim's id.
- A claim whose subject or carried words the earlier claim grounds follows the
  successor while the successor still grounds it; otherwise the message keeps its
  earlier claims (`kept_for_dependents`).
- The held changes that name the earlier claim, its learning tier and the running
  mind's decided relations (when the value is the same) follow the successor.

A retracted value is never made live again:

- A value the message states for the first time, which a correction retracted (or a
  change superseded) after the message was said, is retracted (or superseded) with it.
  Said again after the correction, it stays a new statement, as it was. A claim an
  earlier re-extraction retired counts here under the key it had, with the claim that
  now holds its correction's place.
- A restatement folds into a claim that was retired only later, as it did when it was
  said, and that claim is retired again as it was.
- A link the new reading adds that would retire a claim the owner kept (their answer to
  a contradiction question kept it, they ratified the change it made, or they declined
  a change that would retire it) is not applied (`owner_kept`).
- A held preference change the owner already decided is decided the same way: it is
  never put to them again.

A message the extraction role cannot read now keeps its claims (`not_extracted`).
The reports (`--status`) hold ids and kinds only.

**Yielding and resuming.** The lane starts nothing while a live recall runs or a new
turn's extraction is due, waits for live recall between two messages, and holds a
message's commit while a new turn of the same person is being extracted: that turn
never links to a claim retired under it (the message is read again later). Each
message's commit records it done, with the extraction version that read it, so a
stopped source goes on with its next message after a restart, and a later run keeps
what an earlier one read (a queue an earlier build created gains the version column
when the ledger opens; the checkpoints it held name no version and count for nothing);
when the extractor was upgraded in between, the messages done under the older version
are read again. A source is tried five times (`MAX_ATTEMPTS`, a chosen bound, not a
measured one) before it is left failed with its claims; a missing extraction role is
not an attempt. `COLONYMIND_SOURCE_CLAIMS=off` stops the lane with the rest of
extraction.

What it does not do: it does not re-run the other projections of a source (opinions,
appraisals, commitments), and an opinion resting on a claim whose reading changed
rests on a claim that is no longer current.

`sidecar/tests/test_source_claim_projection.py` exercises actual source ingestion, SQLite transactions, source FTS, projection and context assembly. Controlled extractor outputs make correction, conflict, valid-time, erasure, lease and model-swap checks reproducible. The router check verifies that disabled escalation makes only one provider call.

A separate private neutral run captured eight actual local-model extractions. Replaying those responses through the actual source and context path with explicit reference dates passed five checks: correction, unresolved disagreement, historical validity, effective-date transition and a calendar-day observation. The first run's mistaken March reference for a September correction is retained as a failed assessment. These are bounded behavior checks, not a representative accuracy benchmark, proof of model equivalence, graph performance test or production acceptance. A further check replayed that extraction through current source FTS/projection and the shared selector with the actual LAN reranker: five relevant evidence packets and three unrelated-question abstentions passed. Reranker calls took 110–191 ms. It reused a private threshold from an earlier development set without fitting a new threshold or adding a public default. Before promotion, observe a real ordinary turn produce a completed projection and retrieve its grounded result through the deployed memory provider.
