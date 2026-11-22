# Hybrid recall and calibrated abstention

Automatic context and explicit memory search combine lexical source search in
SQLite with the optional Lance semantic projection. Both use the same canonical
source IDs and enforce participant/session scope before selection. New source
text and corrections remain available while semantic projection is pending.

An absent, incompatible or unavailable semantic index leaves lexical search
available and reports the semantic limitation. A working empty search is distinct
from an unavailable canonical store. There is no other memory store: the Neo4j
graph memory was removed in M8. See [source semantic recall](SOURCE-SEMANTIC-RECALL.md)
for projection identity, model swaps and the shared HTTP contract.

## Returning no useful memory

`COLONYMIND_RECALL_RERANK_MIN_SCORE` is optional and has no global default. Scores
have model-specific meanings. When a cutoff is configured and calibrated, the
reranker also evaluates candidate sets smaller than the requested result count.
Passages below the cutoff are omitted instead of padding the context. Omitted
scores cannot satisfy a configured cutoff. A reranker error retains the existing
fallback behavior; an empty result must not be interpreted as evidence that an
unavailable backend searched successfully.

The cutoff applies only when `COLONYMIND_RECALL_RERANK_CALIBRATION` matches the
SHA-256 configuration fingerprint supplied by the active reranker registration.
The fingerprint uses the provider, model, endpoint, prompt format, candidate
input format, optional weight revision, embedding configuration and optional index generation. Changing
those values invalidates the cutoff. For a registered reranker its inputs, by
name, are `provider`, `model`, `endpoint`, `prompt_style`, `format_version`,
`weights_revision`, `embedding_model`, `embedding_dimensions`,
`index_generation` and `candidate_format`. The optional `query_transform` input
is `identity` or `query-adapter-1:<sha256>`; the adapter caller supplies
its revision in metadata. Omitted and explicit `identity` have exactly the same
fingerprint as before the identity split, preserving existing cutoff records.
A changed adapter revision invalidates the cutoff while document vectors and the
index generation remain compatible. A custom rerank function supplies current
metadata through `RecallSelector(rerank_fn, calibration_metadata=...)`.

An unmatched or absent fingerprint disables the cutoff and marks the returned
rows `mismatch` or `unverified`, with a warning. A matching configuration whose
weight revision is unavailable is explicitly marked
`configuration_verified_weights_unverified`. A configuration stamp cannot detect
an unannounced weight replacement behind the same endpoint/model alias. Operators
must invalidate calibration on such a replacement; setting a model name is not
proof of immutable weights. `COLONYMIND_RERANKER_REVISION` and
`COLONYMIND_RECALL_INDEX_GENERATION` accept known revisions when available.

There is no universal cosine or reranker threshold. A deployment should freeze
representative positive, paraphrased, corrected and no-answer queries, select a
cutoff on its development subset, then test its held-out subset once. Keep that
deployment's calibration values outside generic defaults.

`colonymind.recall_cli.refit_query_adapter_cutoff` fits a record from captured
scores carrying the complete configuration fingerprint, including both adapter factor
scopes. It chooses the cutoff on development queries and checks a disjoint held-out
subset once, including at least 150 no-answer queries. The registered query-adapter
driver persists consumption of that check, including failures. Passing factors and
cutoff promote and roll back as one closure. Owner recall reads that request's pinned
cutoff and transform; the environment values remain the fallback when no learned
closure serves. Provider drift refuses the learned vector lane before it can use an
outdated calibration. Failed qualification keeps the previous closure. The callable
loads no endpoint and re-embeds nothing. Full answer recall and zero false injection
are conservative assumptions; relaxed bars require non-regression against the
previous closure on the same held-out queries.

`colonymind plasticity qualify LEARNER --queries FILE` produces the registered
driver's local qualification capture using the instance's existing index,
embedder, reranker and recall selector. The frozen query file separates
development, held-out, gate and canary queries; captures contain references and
scores without query text. Endpoint overrides and finite query/time budgets are
available. Both sealed recall harnesses accept `--query-adapter` with a portable
global revision directory and record its digest. See the
[query adapter](PLASTICITY-LEARNERS.md#query-adapter-query_adapter) for the file
format, refusal rules and joint adapter/cutoff promotion.

The calibrated candidate format is
`grounded-quotation-bundles-v2-corrections-first-rerank-cut-1500-2000`. Attributed
corrections precede the original evidence in ranking text; all members remain in
the complete output packet. The reranker is sent the first 1,500 characters of each
candidate's ranking text and the first 2,000 characters of the query; the text
shown to the model is not cut. Uncut, a reranker endpoint with a bounded context
refused long recalled messages (an HTTP 400), and long batches outlasted recall's
deadline; either way recall fell back to its candidate order, which skips the
cutoff and injects every candidate. On a copy of a deployed store, with recall
given 10 s, 37% of questions that should get no memory were never scored, so they
all received memory; with the cut, none failed. Changing that format changes the configuration fingerprint even when the
provider and numerical cutoff stay the same. Before promotion, test the intended
cutoff against useful corrections, other useful evidence and genuine no-memory
queries, preserving any earlier failures. A configuration match alone is not
quality qualification, and irrelevant competitors inside useful queries do not
replace whole-query abstention controls.

A build keeps a deployment's calibration only while it keeps this format and
these inputs. Two reference fingerprints check that without a deployment's own
values: `OpenAIAPIRerankerProvider("fixture-reranker")` configured with the
endpoint `http://reranker.test:8000`, no key, and none of
`COLONYMIND_RERANKER_REVISION`, `COLONYMIND_EMBED_MODEL`, `COLONYMIND_EMBED_DIMS`
or `COLONYMIND_RECALL_INDEX_GENERATION` set, fingerprints to
`6a3b3f04d569b15f7ddc576597ca7aaf82c750ebdf39d06f0716f36b5046910a` with the
raw prompt style and
`4ec2ca0198780bb21f3f7af6412d6ac21823360beaef3b507145a6c5af72bdc3` with the
`qwen3` style. A change to either, or to the candidate format, is a change of
what deployments are calibrated on: its release notes say so, this section
names the new values, and each deployment recalibrates on them
(`sidecar/tests/test_recall_calibration_format.py` fails until the code and
this section agree).

Conversation pairs. A reply found by a lane brings the user message it
directly answers, and a user message found whole brings its single direct
reply: the next message of its own source, or a later turn whose supplied
input names it. That second link is written only inside the reply, so a
derived map from each user message to the replies naming it (filled when a
source is written, rebuilt from the sources, and deleted with either source)
finds it without reading the later sources. Every reply the map names is
checked as a found reply's input is (the viewer's scope, the source's
attribution, the exact message and version); no reply, or more than one, is no
pair, and an annotated reply is not attached. When the question is not shown
as a plain quotation (an excerpt of a long message, or words recalled as
claims) the reply is not shown alone. The same question with the same answer
in two conversations is one pair, as a person's repeated words are one
quotation, unless the other conversation's pair brings a correction the one
kept does not (a correction of either message, or of what it depends on, as
correction expansion reads it): then both stay, and no correction is dropped
with its pair. A pair stands where the better ranked of its two messages was
found, with that message's rank: a question found first keeps its place when
its answer comes after every other hit, so the reranker's bounded batch still
holds it.

Known limit: a candidate is ranked by its first 1,500 characters alone. A
conversation pair's ranking text puts the person's message first, so after a
long message the reranker reads little or none of the reply, and a reply that
answers the question can score under the cutoff and be left out. Pairs are
larger than the messages alone, so fewer records fit the character budget,
and a question now brings its reply behind it whenever it is found. A checkpoint's
copy of the reply is a ranking window of its own and stays a candidate whenever
the pair's window does not show its words whole, with the original's
corrections, and gives way to the pair when both are selected ([source semantic
recall](SOURCE-SEMANTIC-RECALL.md)); without such a copy the reply is not
scored on its own words. Ranking each message of a pair in a window of its own
changes what the reranker is sent, so it needs a new candidate format and a
calibration on it.

Select the qualified code and its matching calibration metadata together.
Keeping an older stamp invokes the existing mismatch fallback, which disables
the cutoff; that is not a successful calibrated upgrade. Recovery must likewise
pair the previous representation with its previous stamp while retaining current
source and erasure state. No threshold fitting occurs automatically.

## Qualification

Required observed cases are: a new source found before its vector is available;
the old superseded source excluded; an irrelevant query producing no recalled
passages under a matching calibration; source handles visible in the assembled
context; scoped misses staying scoped; and a changed reranker configuration
invalidating its old cutoff. Existing consent and tool authority remain with
their execution owners.

An earlier development benchmark used actual LAN embedding and reranker models, a
disposable Lance store and a fixture-backed stand-in for the graph memory that has
since been removed. The frozen comparison in `benchmarks/source_recall` now drives
the production `collect_sources` and `select_memory` path directly (M8); its
earlier three-arm reference results have been retired. Produce a new measured
run with the current harness (see that README).

The initial implementation replay retrieved the expected sources but scored
15 of 24 complete behavior cases against its original fixture. Review then found
that the fixture conflated a request for today's footage with a window beginning
the previous day. That aggregate is not a reliable acceptance score. Explicit
calendar, validity and contradiction cases are required. A reranker cutoff does
not replace occurrence-time filtering or canonical contradiction state. This
patch preserves recorded timestamps and existing contradiction counts; it does
not implement those remaining memory semantics.

## Keyword lanes

Two keyword lanes search the same chunks of every source (2,000 characters,
overlapping by 200), each in its own SQLite full-text table:

- `lexical` searches the words as written (`unicode61`: case and Latin
  diacritics folded).
- `stemmed` searches them with English stemming (`porter unicode61`), after
  every run of CJK ideographs, kana or hangul is rewritten as its overlapping
  two-character pieces ("东京出差" is indexed as "东京 京出 出差"), and a question is
  rewritten the same way. There "moved" finds "moving", and a two-character
  word such as "东京" finds "我昨天去了东京出差", which the written lane keeps as
  one token. The stemming is English only: other inflected languages gain
  nothing from it, though their CJK and hangul text gains the pieces.

Both lanes search by the same choice of terms: a question's identifiers first
("p-72", "4.2m", searched as the phrase of their pieces; at most four), then
its words longer than two characters that are not question words ("what",
"when", "remember" and the like), at most 12 terms in all. A question with more
keeps its rarest words: those the fewest indexed chunks hold, counted in the
lane's own index as its tokenizer reads them (folded, and stemmed for the
stemmed lane), kept in the question's order. So a long message searches its
most telling words rather than its first twelve. A word no chunk holds (a
misspelling, or the field names of a format the question asks for) can find
nothing, so it comes after every word some chunk holds and fills only a place
none of them wants. The counts are over the whole store, every person's chunks
(they choose terms; the search keeps the person's scope).
A question of 12 terms or fewer searches exactly what it did before; if the
counts cannot be read, a longer one searches its first 12, as before. The same
choice feeds the claim extractor's search for a message's earlier claims.

The lanes are registered in `LANES` (`colonymind/memory/search.py`) and run at
once in `collect_sources`, where one reciprocal-rank fusion (k = 60) joins them
with the vector lane and each lane's ranks go to the trace. The keyword lanes'
own fusion is the evidence recall has without an embedder, and the order the
fallback without a working reranker alternates with the fused one. A keyword
lane that cannot be read fails the collection as soon as it ends, as the one
keyword lane did, and the lanes still running are cancelled; the vector lane's
failure leaves the keyword evidence.

Each lane scores its matches by bm25 in its full-text table alone and reads the
sources of the best 256 to keep the person's scope, reading further only when
those fall short; the order is the one scope-checked query's order either way.
A stemmed row names its chunk by place (the source, the speaker and the chunk's
index among the source's chunks), and its excerpt is read back from the
canonical source: a row whose indexed text is not the rewrite of the chunk at
its place never reaches the candidates.

The stemmed rows follow their source in SQL, whoever changes it: triggers on
the source table delete a source's rows when the source is deleted or its
messages change (a forget, a partial erasure, the secrets migration), and queue
every new or changed source. A source is indexed when it is written; what is
still queued is indexed when the ledger opens. The first start after an upgrade
builds the table from every canonical source and logs how many sources it
indexed and how long it took; a build cut short resumes on the next start, and
sources an older version wrote after a rollback are indexed on the next start
of this one. `TurnIdempotencyLedger.rebuild_keyword_stems()` rebuilds it from
the canonical sources, with the same rows each time. The table roughly doubles
the keyword index on disk, more for a store that is mostly CJK text.

A third keyword lane, `claim_key`, searches the words of claims' keys (each
claim's subject, predicate and value), written and queried as the stemmed lane
writes and queries messages. A match is the words of the message the claim
quotes, and recall expands it into the claim's grounded bundle: "who is my
employer" reaches `employer: Acme` from "I just started at Acme". It is
registered after `stemmed`, fused and traced with the others, and counts as
keyword evidence ([source claims, Recall and time](SOURCE-CLAIMS.md#recall-and-time)
describes the index and how it follows the claims).

Adding a lane changes the candidates the reranker and the cutoff see, not the
text it reads or the calibration fingerprint: a deployment re-checks its cutoff
on its development queries after the upgrade.

## Recall and time

A question that names a span of time ("what did we talk about last week", "in
March", "3 days ago", "between 20 December 2026 and 24 December 2026") is read
as that span, counted from the turn's own time in the contact's timezone (the
forms, and those left unresolved, are in
[source claims](SOURCE-CLAIMS.md), "Recall and time"). The span sets which
claims are admitted, as before, and adds a lane.

The `window` lane reads the person's sources in the span straight from the
ledger, so a message whose words and meaning the other lanes do not match still
reaches the reranker: the week's messages for "what did we talk about last
week" share no word with it. A source is in the span when it occurred there
(its occurrence time, or when it was recorded if that is unknown), and a message
is when one of its claims' event times falls there (a message sent today about
last Tuesday). The lane orders its candidates by the question's other words
matched within the span (the keyword lane's query, the date words left out),
then by the vector lane's rank for the in-span messages it already returned,
then newest first. It divides the span into 8 equal parts, reads each part's
newest messages and events on their own (at most 15 each), and takes one from
each part in turn, so a busy day cannot fill it; it returns at most 15, and one
reciprocal-rank fusion joins it with the other lanes. A message found through
its claim is placed by its event's time, not by when it was sent. A span open to
the past ("before X") begins at the earliest message or event the lane finds in
it, since equal parts of all time would put everything in the last one. Those
numbers and that rule are the builders' choices, not measured optima.

It runs only when the question's time resolves to a span, after the other
lanes (it reads the vector lane's ranks) and where the time is read, once the
contact's timezone is known (`select_memory`); it is registered in `LANES` as a
timed lane and traced there. Its scope is the keyword lanes' own: the person's
sources and this session's session-scoped ones, never another person's, never
a source whose attribution was invalidated. A failure of the lane leaves the
other lanes' evidence. It only adds candidates: the reranker and its cutoff
judge them like any other, and without a reranker they take their fused place.

The lane reads an index of each person's sources by occurrence time, and
indexes of claims by event time; the first start after the upgrade builds them
and logs how long it took. Its keyword order ranks the best 2,000 matches in the
full-text table alone first, as the keyword lane does; when they hold too few
in the span, it reads every match within the person's scope and the span, so
better matches from other people or other times never hide the span's own.
Claims are read by event time, newest first, with the person's scope, a
retraction and the overlap with the span checked before a claim counts, so
claims the person cannot see never use up what the lane reads; an event period
is looked for up to 400 days before the span. There is no windowed
vector query: the vector index holds when a row was indexed, not when its
message occurred.

An explicit search can name its own time. `POST /v1/host/memory/search` takes:

- `window`: `{"start": ..., "end": ...}`, timezone-aware ISO timestamps, the
  end exclusive; a window whose end is not after its start is refused with 422.
  It is the question's span whatever the words say: it adds the `window` lane
  and sets which claims are admitted (by when an event happened when the
  question asks what happened, was seen or recorded; otherwise by when a fact
  held). It never removes a quotation another lane found, so a wrong window
  cannot hide evidence: in an event window a quotation from outside it is kept
  and labelled `outside_requested_window`. A found message that recall reads
  through its claim (the claim covers its words) is kept too when the window
  does not admit the claim: it is offered as a question with no date would see
  it, labelled `outside_requested_window` when the claim's time is known and
  outside the window, `unknown` when nothing dates it. A retracted or
  superseded claim's words stay out, as without a window. A span read from the
  words keeps its existing rule: an event span drops quotations from outside
  it.
- `as_of`: a timezone-aware timestamp, clamped to no later than now: the time
  relative dates and the current view count from ("yesterday" is the day before
  it; a current fact is the one valid then). It is not a view of what was
  believed then: everything recorded since, corrections included, still counts.

The trace names the lane `window`, and `colonymind recall --explain` calls it the
date window.

## One selection path for turn context

`/context/assemble` combines scoped lexical and semantic source hits,
source-claim expansions and media descriptions (the retired contact estimates no
longer enter it; see [memory quality](MEMORY-QUALITY.md)).
Canonical source search independently enforces contact/session scope and
source-message erasure. A partially redacted turn can retain unrelated quotations.

Scores within `COLONYMIND_RECALL_RERANK_TIE_MARGIN` of each other can be made a
tie that the fused retrieval order breaks: each place in the packet goes to the
earliest fused candidate within the margin of the best remaining score, so no
candidate goes ahead of one that scored more than the margin higher. A person's
own words, the agent's restatement of them and tool output repeating the same
facts all answer a question about them, and a cross-encoder can score them
within a hair of each other (one candidate's score also moves a little between
two calls); sorting on those differences can put the restatement first, and the
character budget then drops the person's words. The margin reorders near ties
only. It changes no score, no candidate text and no cutoff decision, so it is
not part of the configuration fingerprint. What difference is a tie depends on
the scale of a deployment's scores, and a verified fingerprint establishes the
configuration, not that scale: on a reranker scoring .001 for weak candidates
and .03 for the answer, a margin chosen elsewhere would merge a thirtyfold
difference. There is no margin by default. A deployment sets one it validated on
its own development queries with its cutoff (the ties among candidates that
answer, against the smallest gap to a clear preference), and it applies only
while a verified cutoff does; otherwise selection keeps the reranker's order.
Setting 0 turns it off.

Canonical candidates feed one rank-fusion and reranking pass. The calibrated
cutoff can reject every candidate. The memory is one `colonymind-memory` section.
There is no separate conversation-evidence injection. Quotations remain evidence
of what was said, while claims retain their distinct interpretation and
uncertainty.

A context packet holds at most `context.recall_items` records (owner-adjustable default 20),
whether or not a verified reranker cutoff chose them. The default matches the independent
20-candidate rerank bound. Candidates are packed in
rank order within the character budget, with the existing skip and shorten
rules. A missed rerank keeps the five-record fallback
ceiling; `COLONYMIND_RECALL_FALLBACK_LIMIT` can shorten it. Exact media locators
share the record and character caps. Explicit searches keep their requested
limit (five by default, up to 20).

A turn's recall sends the inline reranker at most 20 candidates. This bounds
model work as producers expand, while leaving room to select useful passages
below the first five. The bound is its own, not a multiple of the packet's
limit, so raising the context record cap does not widen it. An explicit search
sends four times its own limit, as before (20 for the default five). Up to the
bound, every candidate is sent. Past it, they are chosen round robin across
their sessions: each session in the order of its strongest candidate, one
candidate per session each round, strongest first within it, so the overlapping
chunks of one long conversation cannot fill the batch and leave every other
conversation unread. A projected assertion bundle is grouped by its key, not
by a conversation, and one long conversation can state many properties: it
names the session of the source it is anchored in, so it counts as that
conversation. A candidate with no session of its own counts as the session of
its source, as the excerpts, pairs and complete messages read from that source
say, and otherwise as that source's. The spread only chooses: it
changes no score, no candidate text and no cutoff decision, and the reranker
still decides; it can send a weaker candidate of another session in place of a
crowded session's stronger one. The existing timeout still applies. On
successful active reranking, only that submitted set competes for the packet;
unsubmitted rank-fusion scores are not compared with cross-encoder scores.
Disabled, shadow and failed reranking preserve the full original candidate set
in its fused order. This can exclude relevant evidence the batch did not hold.
Evaluate source retention alongside latency on the deployment's corpus; a
smaller batch does not establish a relevance cutoff or solve unrelated-memory
injection.

Plain attributed quotations that exactly repeat the current request are
supplementary: selection places them after independent evidence before bounded
reranking (the spread sends them only after every other candidate) and final
packing. Original bytes and source records are retained;
qualified assertion histories, annotations and uncertain attribution are exempt.
This reduces displacement by repeated questions, including while a correction
awaits claim processing. It does not guarantee primary-source coverage when
many distinct assistant retellings compete, or recover sources missed during
candidate retrieval. The packet's limits and character budget are unchanged.

The current session's own quotations and conversation pairs stay candidates:
whether the host still shows them verbatim is not known across its agent
rebuilds and compactions, and leaving them out on a wrong guess loses the only
verbatim evidence for turns the host has summarised away.

An explicit search's combined rendered budget is 4,000 characters. A turn's packet
takes what the context budget has past the 6,000 characters the other sections
keep: `context.budget_chars` (default 16,000), or the host's inline allowance when
that is smaller, less 6,000, and never less than 4,000. That is 10,000 characters
at the defaults and 4,000 under an allowance of 10,000 or less.
`COLONYMIND_RECALL_CONTEXT_MAX_CHARS` names one budget for both, up to 24,000; zero
suppresses this packet.
When the host names its inline allowance, a turn's packet also stays within that
allowance less the items kept whole beside it (see Per-turn context selection
below). This is a character limit, not an asserted token count. Shortened excerpts carry
`excerpt_truncated=true`, and source bytes remain intact in the source store.
Records retain `kind=belief` or `kind=source_quote`, source/turn handles, speaker
role, and occurrence/ingestion times when available. Quoted content is explicitly
evidence rather than instructions or an accepted belief.

Reranker failure uses one shared bounded fallback and marks returned records
`rerank_status=unavailable`; it does not promise calibrated abstention during an
outage. It fills the packet up to its own limit unless a deployment calibrated a
smaller bound (`COLONYMIND_RECALL_FALLBACK_LIMIT`; see Recall latency below).
Unconfigured reranking retains rank-fusion fallback. This budget
covers the mixed memory packet, not other existing context sections.

## Per-turn context selection

The memory packet is one lane of the per-turn context, and it arrives judged:
recall retrieved wide (lexical, vector and claim candidates), the calibrated
reranker scored each passage against the incoming message with its whole
ranking text, and the packet was fitted into its own budget
(above: 10,000 characters at the defaults). It is injected as recall
selected it: never judged again, cut or charged to the selection budget. With
selection on (`COLONYMIND_CONTEXT_SELECTION`, default `on`), `/context/assemble`
pools the items of every other lane it assembled (pending commitments, recorded
views, lessons, the relationship and communication-landscape lines, observed
work and the other sections), has the same reranker score each against the
incoming message, and keeps the best-scoring items within one character budget
(`COLONYMIND_CONTEXT_SELECTION_CHARS`, default 3,000 characters, counted as the
host renders them: each section's `## <title>` heading and the blank line
between sections included). Kept items stay in their sections, in their assembled order and words;
a section keeps its header while any of its items remains, and its citations
shrink to the sources its remaining text names. The judge reads each item's
words with its section title, without identifiers, hashes or timestamps, at
most 600 characters of it; at most `COLONYMIND_CONTEXT_SELECTION_CANDIDATES` (48)
items are judged. Above that cap every candidate is first ranked against the
message by the words it shares with it (rarer words weigh more, a five-letter
stem absorbs inflections), lane priority breaking ties, so the cap never drops
the item the message asks about in favour of a higher-priority lane.

Pinned items are never judged and never dropped: Current Time, Expected replies
(the open asks), the owner's communication preferences and priority
corrections, the corrections to earlier context below, and every commitment
that is overdue or due within `COLONYMIND_CONTEXT_DUE_SOON_HOURS` (24). A turn
whose other lanes already fit is returned without a reranker call. The memory
is recalled while the other lanes are assembled; the judge runs once recall has
finished, because the two share the reranker and recall's own deadline must
never be spent waiting behind the judge's call. With no
reranker configured, or when it fails, answers late
(`COLONYMIND_CONTEXT_SELECTION_TIMEOUT_MS`, default 2,000, and never past what
the turn's budget leaves), is still answering recall's call (the judge skips
at once, reason `busy`) or returns an
incomplete, duplicate or non-finite score set, the assembled context is used
(as far as the host's inline allowance takes it: the pending commitments first, then
the other items in assembled order) and the fallback is logged (a failure at most once per five minutes
at warning level). `COLONYMIND_CONTEXT_SELECTION_MIN_SCORE` (default unset)
drops judged items below a score even when the budget has room; like the recall
cutoff, a threshold belongs to the deployment's reranker and corpus.

The host's inline allowance bounds the whole injection. A host names it with
`inline_chars` on the request: the characters it puts inline, counted as above.
The Hermes provider sends its spill cap (`hooks.output_spill.max_chars`,
10,000 by default; Hermes moves a longer prefetch to a file and shows a
preview) less the room its own Current Time block takes. The memory packet and
the pinned items (the corrections to earlier context included) are counted
first and kept whole; the optional items fit into
what is left, within the selection budget when judged and, when the judge is
unavailable, the pending commitments first and then the other items in the
order they were assembled.

The memory packet itself stays within the allowance less the pinned items.
Recall packs within its own budget while the other lanes are assembled, and a
packet could fill that budget beyond a shorter host allowance. Once the context is assembled, the corrections to earlier context and
the superseded-value annotations included, the pinned items are counted as the
selection counts them, and when the memory and they exceed the allowance, what
recall's packer was offered (every candidate left after ranking, checked again
as recall checks its packet) is packed again within what they leave: in rank
order, with the packer's rules and the packet's count limit unchanged, and its
citations are the sources it then shows. Packing from the offered candidates
rather than from recall's packet matters when a record shrinks: a long procedure
that recall's budget kept whole becomes a pointer to its source in a smaller
allowance, and the short answer recall's budget had left out behind it takes the
room it frees. A record of recall's packet that no longer fits is left out (the
trace says `over_host_allowance`); one recall had already left out keeps its
reason. With the annotation on, the packet is annotated as the assembled
memory was, and it is the annotated packet that has to fit: one that fits only
without its notes is packed within less. The claims the owner's turn is
recorded as served are the ones the packet then shows. When not one record fits beside the pinned items, the
section is held back whole, citations included: never its heading and the
packet's first line without a record (the block fit keeps a head of any block
it admits), and never a packet over the allowance with the resident blocks off.
The budget report lists it among `held_back`, the `Held back this turn` section
names it in room kept for it (a budget too small for those words still names
it on the report), and the trace says `held_back`.

When the memory and the pinned items alone still exceed the allowance they are
kept whole, nothing else is added and a warning is logged (at most once per
five minutes); the host's cap then has to hold them.
A Hermes host moves a prefetch longer than its cap to a file as a whole: the
provider cannot keep part of it inline, so a deployment whose owner-critical
items can exceed 10,000 characters raises `hooks.output_spill.max_chars`.
A request without `inline_chars` is bounded by the selection budget alone.

Why the memory packet is not re-judged. The first version pooled the memory
items with the other lanes. On a copy of a deployed store (552 per-turn probe
requests over owner claims, from another session and from the claim's own
session), the owner turn that stated a claim was in 35 assembled contexts and
dropped from the selected one: 33 by the budget and 2 by citation trimming. A
recalled owner message is about 1,100 characters (up to 3,600), so a
3,000-character pool holds two; a message longer than the whole budget was
skipped while items the judge scored zero filled it; and the judge's short
reading scored near zero passages recall had ranked first (voice transcripts,
questions that leave the value out). Replayed on the same answers, keeping the
packet as recall selected it and judging the other lanes left every probe rank
as assembled, while the mean context went from 9,800 to 6,700 characters (5,700
with a 2,000-character budget); the first version kept 3,400 characters and
lost 12 to 18 points of the probes' mean reciprocal rank on the transcript
family. This is answer retention in the context, not a model-in-the-loop
accuracy run.

## Recall latency

A host waits a bounded time for `/context/assemble` and goes on without a later
answer (Hermes: 8 s). Recall's own limits (a query-embedding limit with
retries under it, the rerank deadline and the judge's) could add up past that
wait, and its misses skipped the calibrated cutoff. A reranker server may finish
a call its client gave up on, so a missed deadline saves no work and makes the
next call wait behind it. Recall now runs within one budget per turn and one
rerank call at a time.

The turn budget (`memory/turn_budget.py`) starts when the request arrives and is
derived from the host's wait: `wait_ms` on the request (the Hermes provider
sends what is left of Hermes' prefetch bound, less its 0.5 s margin, when it
asks), else
`COLONYMIND_CONTEXT_HOST_WAIT_MS` (default 7,500). The query embedding gets one
attempt of `COLONYMIND_RECALL_EMBED_TIMEOUT_MS` (default 1,500; explicit
searches too). Recall's rerank gets the lesser of
`COLONYMIND_RECALL_RERANK_TIMEOUT_MS` and what is left of the wait less a
reserve: the judge's deadline (when selection is on) plus 500 ms, at most a
third of the wait, so recall comes before the judge. The judge gets the lesser
of its own deadline and what is left less 500 ms. With nothing left no call
starts. With a hung embedder and a hung reranker the route still answers
within the host's wait: one embedding attempt, the rerank deadline, and a judge
that skips the busy reranker.

One rerank call at a time (`memory/rerank_gate.py`): recall and the judge reach
the reranker through one gate per reranker, process-wide. A call starts only on
an idle reranker. Recall's call keeps the gate until the server answers, whether
or not recall still waits (at most `COLONYMIND_RERANK_HOLD_LIMIT_MS`, default
30,000), so the judge after it skips instead of waiting behind it. The judge's
call keeps the gate only while the judge waits: the judge has no use for a late
answer, so when its deadline passes or it is cancelled its call is cancelled and
the gate is free at once, and the next turn's recall reranks instead of skipping
behind it. Single flight on the server lasts only that long: the gate knows when
an answer arrives, not when a server stops working on an abandoned call, so once
the gate is freed (past the hold limit, or once the judge gave up) the next call
may reach a server still computing the first.
A call that finds the gate held is skipped at once, never queued
behind work the server has not finished. Concurrent turns therefore do not
share a busy reranker: the later one falls back. An answer that arrives after
recall's deadline is used while the turn's other lanes are still being
assembled, since it costs the turn nothing. The explicit
`/v1/host/memory/rerank` API is not gated.

The query embedding: a request the endpoint refuses (4xx, such as a query past
the model's context) is not retried, and transient failures of background
batches keep their three attempts. The query is embedded while the keyword
search runs. With its instruction prefix it is cut to what
`COLONYMIND_EMBED_QUERY_MAX_TOKENS` admits (default 4,096; a deployment sets its
endpoint's own input limit), counted in UTF-8 bytes: a tokenizer emits at most
one token per byte of its input plus a few markers of its own (16 are left for
them), so no script can take the query past the limit, where a character count
could (one character can be four bytes and several tokens). The cut never
splits a character, and an ASCII query keeps about its first 4,000 characters,
the keyword search's own bound. The query has a small LRU cache of
its own (`COLONYMIND_EMBED_QUERY_CACHE_SIZE`, default 128). Background batches
are not cached. Remote providers keep one pooled HTTP client per event loop,
whose idle connections expire after `COLONYMIND_HTTP_KEEPALIVE_MS` (default
4,000, shorter than a typical server's or proxy's own idle timeout). A pool is
closed within its loop's shutdown (`asyncio.run` and every shutdown that
finalizes the loop's async generators), so no pool keeps an ended loop or its
sockets; a loop closed without that shutdown has its pool dropped the next time
the provider is used from another loop. Background source
indexing starts no batch while a live recall runs, waiting up to 8 s and then
sending 4 chunks instead of 16 (docs/SOURCE-SEMANTIC-RECALL.md).

A missed rerank (reasons `no_reranker`, `budget`, `busy`, `timeout`, `error`,
`invalid`) is counted every time in health's `recall` block, turns apart from
explicit searches and the mind's packets, with the fallback records injected,
the late answers used and the judge's outcomes. The rate-limited warning stays;
the count is exact, whatever a miss injected. It equals the number of turns
whose memory shows `rerank_status=unavailable` only while every missed turn
injects at least one reranked candidate: a missed turn whose fallback injects
none (a bound of 0, or exact media locators, which come first and are never
reranked, filling the packet) is counted and shows no such marker.

A missed rerank injects what it did before: the fallback's fused order up to
the packet's own limit (the lesser of `context.recall_items` and
`MEMORY_PACKET_LIMIT`, 5, for context; an explicit search's own limit). A smaller bound trades answers the fallback holds for
fewer injections on questions that should get no memory, and only a
calibration on a deployment's own positives and negatives, answer retention
included, can set that trade. A deployment that has one sets
`COLONYMIND_RECALL_FALLBACK_LIMIT`, which only ever lowers the packet's limit (0
injects nothing), in a turn's context and an explicit search alike. Health's
`recall` block reports a turn's bound (`fallback_limit`) and whether a
deployment set it (`fallback_limit_set`). Unlimited-packing counters have been removed.

None of this changes the candidate format, so a calibrated cutoff and its
fingerprint stay valid.

## Tracing a recall

A recall can say why it kept or left out each candidate. `POST /v1/host/memory/search`
takes `"trace": true` and returns a `trace` beside the packet; `POST
/v1/host/context/assemble` takes it too, from a credential that manages the
instance (the owner's side of the API). A credential that does not manage the
instance may trace a search of its own person only (its viewer, or its own
identity); any other request for a trace is refused with 403
`trace_not_granted`, and development mode never gets one. From the command line:

```
colonymind recall "<question>" --as <person> [--session <id>] --explain
```

asks the running sidecar with the instance key and prints one line per
candidate, for example `found by keyword (rank 3) and vector (rank 1); fused 1;
reranked 0.81 (calibrated cutoff 0.67, passed); in the packet`. Without
`--explain` it prints the packet the search returns (`--limit` records at most,
5 by default, an explicit search's own; context uses `context.recall_items`); `--json` prints the answer as it came.

Each row of the trace is one candidate:

- `turn_id` and `message_hash`: the source and the message it came from (for a
  candidate that carries several messages, such as claims or a conversation
  pair, the first of them);
- `candidate` and `form`: its id in this recall (a kind and a digest) and the
  form it was ranked in (`quotation`, `claims`, `conversation`, `correction`,
  `media`, `media_name`, `belief`);
- `lanes`: its rank in each retrieval lane that found it: `lexical`, `stemmed`,
  `vector` and `window` over the messages (see Keyword lanes and Recall and
  time), `claim_key` over the claims' keys (see Keyword lanes), `vector_media` and
  `lexical_media` over image
  descriptions (an image lane's rank counts for the image it found, so two
  images of one message keep their own ranks, never for the words of the
  message it came with);
- `fused_rank`: its place in the fused order the reranker is offered;
- `submitted` and `rerank_score`: whether the reranker was sent it (counted
  when the call starts, so a rerank that then timed out, failed or answered
  invalidly still says what it was sent), and its score when the rerank scored;
- `exit`: null when it is in the packet, else one reason, in the order of the
  stages that leave a candidate out:
  - `no_lane`: no lane carried it to ranking: not found, or found and then left
    out before ranking (a repeat of words already ranked, outside the
    question's time, or a reply that is shown only beside its input);
  - `below_submission_bound`: the reranker was sent at most the submission
    bound of candidates (20 for a turn, four times the limit for an explicit
    search), and this one was below it in the fused order;
  - `over_session_share`: it was within the submission bound in the fused
    order, and the batch, spread across sessions, went to other sessions'
    candidates and its own session's stronger ones;
  - `below_cutoff`: it scored below the calibrated cutoff. This is only ever
    reported while a verified cutoff applied: with a calibration that does not
    match, no candidate is labelled below it;
  - `over_packet_limit`: the packet was full (a packet with a count limit:
    an explicit search's, or a turn's without a verified cutoff);
  - `over_char_budget`: it did not fit the character budget;
  - `over_host_allowance`: it fitted recall's budget, not the host's inline
    allowance less the items kept whole beside the memory (context assembly,
    when the host names its allowance);
  - `removed_by_recheck`: its source or its corrections changed while it was
    ranked;
  - `erased`: something it came from was erased during the recall: a source it
    names, or a message it carries (erasing a message also takes its copies out
    of the checkpoints that keep other messages, so a source that stays can lose
    one). The row stays, without its ids, and `selected` and `shown_by` no
    longer name it.

A message (or an image) a lane found that no candidate carried to ranking has a
row of its own, with `exit` `no_lane` and no candidate. A candidate whose words are in the
packet through another one (a checkpoint's copy of a message that a
conversation pair shows whole) has a null `exit` and names that candidate in
`shown_by`.

Around the rows, `lanes` counts what each lane returned; `rerank` gives the
mode and what the reranker did (`off`, `not_needed` when every candidate fit
and no cutoff applied, `scored`, `shadow`, or `missed` with the miss reasons
health's `recall` block counts), the submission bound and how many candidates
it was sent (none when no call started: `no_reranker`, `busy`, `budget`);
`calibration` gives the configured cutoff's check and the verified cutoff;
`packet` the count limit, recall's
character budget and the count; `selected` the ids in
the packet, in order; and, for context assembly, `section` says whether the
recalled section reached the returned context as recalled (`in_context`),
`shortened` or `held_back`. At most 256 rows are returned; `rows_omitted`
counts the rest.

The rules:

- The trace carries ids, digests, lane names, ranks, scores and closed-set
  values: never text, never the question. Its schema has no field that could
  hold text.
- It is built for one response and never stored.
- It is limited to the caller's scope. Every id in it is read back against the
  ledger in the recall's own scope when the trace is reported: a candidate keeps
  its ids only while every source it names is readable there and every message
  it carries is still in its source, so a source or a message erased meanwhile
  leaves no id.
- The packet is the same with the trace on and off, and a response nobody
  asked to trace has no `trace` key. With the trace off the recall is handed a
  no-op collector and does no work for it.
- It changes no candidate, score or cutoff: the candidate format and the
  calibration fingerprint are unchanged.

Each lane records its ranks where its hits join the candidates: the message
lanes where they are fused (the `LANES` of `collect_sources` in
`colonymind/memory/search.py`; a timed lane, `window`, in `select_memory`),
the image keyword lane where recall reads it (`select_memory`). A lane whose
hits join anywhere without recording them is invisible to the trace. A selector
other than the built-in one is traced only if it accepts a `trace` argument.

## Superseded values

This ships off: it runs only with `COLONYMIND_CONTEXT_ANNOTATION` set to `on`
(`1`, `true` or `yes` too), which a fresh install leaves unset. Without it, no
line is annotated and no earlier value is corrected.

With it, a line of any section that states a value the record has since
superseded keeps its words and gains the current value: `[superseded: now "<value>" since
<date>]`, `corrected to` for a correction, `rescheduled to` for a commitment
deadline. The records are the scope's changed and corrected source claims
(followed to the end of the chain, however long, and only to a claim read and
found superseded by nothing; a cycle, a chain past 4,096 claims, or an erased or
unattributed successor asserts nothing) and the listed commitments' previous deadlines. A line is a record's
only by identity: it carries the commitment's `id=`, or the claim's id or the
`turn:` of the source that stated it. Shared subject words, or the same value on
another record's line, never count. Values are matched in the line's words (a
JSON record read as its string values, identifiers and timestamps left out, and
every JSON escape read as the character it stands for, so `Caf\u00e9` is
`Café`), so
a value of any length counts on its own record's line and a "42" inside a
timestamp does not. A record's line counts when it carries the
old value without the current one, or a note whose value the record has since
replaced (a deadline moved twice), or, on a commitment line, a due other than
the current one. A line gains one note per superseded value it shows, with no
cap, and a note answers only for the value it corrects: a line with notes for
two of three stale values still counts as stale. The paired report's
dead-value secondary counts the same way per artifact: a note answers only for
the artifact whose expected value it states.

The host replays earlier turns' context as it was. The sidecar keeps, per
conversation and in memory only, what it served; when a value served earlier has
been superseded since, the new turn carries a pinned "Corrections to earlier
context" section naming the record, the old value and the current one. A
correction is owed while the last line of that record the conversation was
served showed a replaced value; once a correction (or a line showing the
current value) has been served after it, it is delivered and not repeated. At
most eight are listed per turn, the most recently served stale values first;
the rest follow in the next turns. That record is
bounded, per process and lost on restart, so after a restart only the current
turn's lines are annotated.

Integrated regressions cover mixed abstention, one budget/section, selection
without any graph import, visibility before model input, failure fallback,
excerpt preservation and the shared result limit.
Six neutral checks with the current LAN reranker also passed, including three
no-answer queries; these are integration checks, not a fresh calibration fit or
a representative holdout benchmark. A deployment must still qualify quotations
and its actual corpus before treating its cutoff as measured for that corpus.
