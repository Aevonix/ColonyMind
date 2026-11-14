# Memory quality

Ordinary turns are evidence first. The canonical source ledger retains their
words, speaker, time, scope and media references. Its lexical and semantic indexes
make that evidence searchable even when no durable assertion is extracted.

There is one automatic assertion extractor. Each accepted assertion requires an
exact quotation, a supported subject/value, a useful memory category and a short
reason it could help later. Categories cover preferences, personal context,
relationships, decisions, procedures and significant events. The use judgment
is stored as `memory_quality` and explicitly labeled unverified. It is neither a
truth score nor permission to act. Corrections, changes and conflicting reports
keep the existing source and time rules.

Extraction and semantic admission review use the same criteria for future
usefulness. Review checks whether the complete subject, relation and value
express a supported property, and assesses the extractor's recall reason. A
copied fragment or an imagined future task is insufficient. This uses the
existing review call; it remains a fallible model judgment.

The extractor stores a procedure's selected exact evidence passage as its value,
within the existing 500-character evidence limit. It asks for conditions and
subsequent steps together, but extraction and admission review can still split
one instruction into several claims. An individual claim does not establish
conditional completeness. The model does not generate a second version of the
stored instructions. A standing preference can likewise use the explicit
`representation="preference"` form: its exact evidence, up to 500 characters,
is a quoted statement including attribution, conditions and exceptions. A
nonliteral generated preference value is discarded in favor of this quoted form
when it cannot be grounded as a supported partial revision. Existing literal
compact preferences and other factual values retain their 160-character limit.
Stored claims remain readable and are not rewritten.

Quoted preferences always require the existing semantic admission review, even
when they quote a whole message; they cannot use the episode review bypass.
Review judges the chosen preference and its scope, rather than treating the
whole quotation as a scalar value. Recall retains distinct quoted witnesses
without inferring equality or contradiction from their wording. Its comparison
marker preserves that uncertainty beside any compact-value comparisons.
Explicit reviewed corrections and changes still select their exact predecessor
and govern its lifecycle, including when normalized wording happens to match.
Quoted conditions remain available even when their applicability cannot be
resolved to a date. Recall does not label them as currently satisfied. A future
change preserves its predecessor until the recorded boundary; an unresolved
effective date does not silently become the message's ingestion time.

On a binding with verified [structured-output support](FUNCTION-ROUTING.md), a
source of at most 500 characters requests its complete message as constrained
evidence, including trailing dates, reporter wording and qualifications.
Local quotation validation checks exact spans; it does not independently prove
that the provider honored the complete-message constraint.
Longer messages retain bounded exact-span selection. Each request owns its
schema; source text never becomes shared routing configuration. Full-message
evidence also includes neighboring clauses, and the existing sensitive-evidence
filter can reject the whole short passage. The original source remains retained.

Routine status, build/test progress, acknowledgments, boilerplate, fictional or
hypothetical examples and debugging output should remain source history. Facts
inside a narrative do not become assertions about the actual world. Real props,
project decisions, reported events and conditional procedures can still qualify
when their source supports that interpretation. Short and mutable
facts can still matter: a key location, an appointment or a reusable repair
procedure should not disappear merely because it can change.

Canonical ingestion no longer duplicates every exchange as an episodic
summary. Automatic ToM fact extraction is also removed: guessing what a contact
accepted from an assistant is a separate task from learning a supported assertion.
Contact affect comes from the existing appraisal call.

The shared contact facts are retired, with their APIs (`/v1/host/mind/facts*`)
and their place in recall. No writer was left, and recall only ever used the
facts linked to a current source, whose sources it already selects directly.
The memory recalled into per-turn context (`/v1/host/context/assemble`) and by
`POST /v1/host/memory/search` now comes from the canonical source ledger alone:
source evidence, claims, annotations, media and corrections, selected under one
character budget. A fact that mattered is
recalled from the message it came from, and a fact the owner wants kept is
told to the agent again, which records it as a source.

The facts themselves are archived, never deleted. `colonymind upgrade` moves
`colonymind-facts.db` into `<instance>/backups/<stamp>/retired/` and keeps a copy
beside `retired/`. It is an ordinary SQLite file with one table,
`shared_facts` (`id`, `contact_id`, `fact`, `source`, `confidence`, `created_at`,
`expires_at`, `metadata`, `source_lineage_json`). Inspect it read-only, for
example with the `sqlite3` shell:

```bash
sqlite3 -readonly "<instance>/backups/<stamp>/retired/colonymind-facts.db" \
  "SELECT contact_id, fact, source, created_at FROM shared_facts ORDER BY created_at"
```

and export it with `-csv -header` (or `.mode json` for JSON) and a redirect to a
file of your choice. Without the `sqlite3` shell, Python's own module does the
same:

```bash
python3 -c 'import csv, sqlite3, sys; rows = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True).execute("SELECT contact_id, fact, source, created_at FROM shared_facts"); out = csv.writer(sys.stdout); out.writerow([c[0] for c in rows.description]); out.writerows(rows)' \
  "<instance>/backups/<stamp>/retired/colonymind-facts.db"
```

The rows carry what they recorded: their historical source
and metadata do not reliably tell owner curation from automated extraction, and
no curation or canonical source is invented for them. The release before the
prune still reads the store, so rolling back (see the
[install guide](INSTALL.md#update)) brings the facts back into use.

Persistent extraction consumers use the provider's completed final answer.
Reasoning-only and truncated responses are not saved as assertions, affect
or image descriptions. Image descriptions remain fallible derived
evidence tied to the original asset; they have a 160-word limit and a separate
output budget for reasoning models. The job status exposes failed attempts so
an unavailable or unsuitable model is distinguishable from an empty result.

Repeated reports keep their distinct source identities through lexical retrieval,
so date filtering and correction expansion can select the right occurrence.
Shortened lexical and semantic excerpts are marked as incomplete. A successful
rerank must score every submitted candidate; partial or malformed output keeps
the original ordering with an unavailable status instead of mixing incompatible
scores. Selection remains bounded by the common item and character budgets.

Lexical hits are hydrated against the owning canonical message before receiving
its message hash, modality and uncertainty. A typed message in a mixed text/audio
checkpoint does not inherit a different message's transcript label. Equal rendered
text from distinct messages retains both message identities through candidate
fusion and correction expansion. An index excerpt absent from the current source
is omitted; retrieval does not make a stale projection authoritative. This is a
lineage guarantee, not independent verification of the quoted words.

A procedure extracted from part of a message recalls that complete current
message as one quoted evidence unit, including unclaimed conditions. Ranking
and character packing cannot select its separate property fragments. If the
unit does not fit, recall supplies a full-source prerequisite when space permits,
or omits it. Opening one property's assertion history alone may omit sibling
conditions. If a source claim was corrected, retracted, expired or conflicts
with another current assertion, recall requires source and history inspection
instead of reviving obsolete instructions from the original message. Attributed
source corrections retain their existing indivisible evidence and erasure links.

This preserves message completeness, not the correctness of the procedure or
instructions in other messages. Text from retained audio remains an unverified
machine transcript with its original source ownership; raw audio is not injected
into a text-only request.

The shared memory header preserves the distinction between actual observations,
reports, fiction and hypotheses. On qualified native Hermes, the existing request
middleware replaces the recognized outer authoritative-memory note with this
evidence contract. It preserves the person's input and the supplied source packet.
An unknown wrapper stays unchanged and requires renewed native qualification.

## Evaluation and limits

The neutral cases in `sidecar/tests/fixtures/memory_quality_cases.json` exercise
ten useful and ten unwanted inputs. Expected outcomes belong to the evaluator,
never to extraction prompts. Compare unwanted promotion and missed useful
information together, then inspect quotations and perform cross-session recall.
These cases are a small regression set, not a population estimate or a guarantee
for every model. Use deployment examples retained locally to extend evaluation
when ordinary behavior reveals a failure.

Structural validation cannot prove semantic usefulness. A model can still
misclassify an item or supply an unhelpful reason. Extraction rejection does not
delete its source, and retrieval of a source does not certify its contents.
The ordinary claim-status API exposes extraction version, errors and claim
counts; accepted claim records retain their quality judgment and provenance.

Claim-job status also exposes bounded `diagnostics` once its current attempt
finishes. It distinguishes an empty model array from candidates rejected by
validation, with candidate/accepted/rejected counts and fixed rejection reasons.
Each rejected candidate records its first failing check; these counts do not
judge whether an empty result was useful. Malformed arrays remain failed jobs.
Counts aggregate the attempt's completed text extractions. `accepted_count`
means validated candidates; the existing `claim_count` reports persisted claims,
which can differ after deduplication or partial work. `last_model_provenance`
identifies the last completed extraction response, including a response that
produced zero claims. It does not describe every response in a multi-message job.
No quotations, rejected values or raw model drafts enter this diagnostic record.
Historical jobs and attempts run by predecessor workers without these
measurements return `diagnostics: null`; an older attempt's counts are not reused.
The record lives on the existing job row, follows its lease, is replaced by the
next finished attempt and is removed with source erasure.

New claim jobs use `source-claims-v11`: after the existing quote, subject, value
and date checks, one batched `source_claim_review` request uses the configured
`judging` role to assess whether each proposal preserves the source's relation,
attribution, negation, modality and memory category. Exact required proposal
keys avoid asking the model to generate matching indices. Runtime validation
still rejects missing, duplicate or invalid decisions, including on bindings
that do not declare strict JSON schema support. There is no review call for an
empty validated proposal set. Useful reports, temporary knowledge, standing
conditional preferences and actual reusable procedures remain eligible.

Whole-message text episodes bypass this second model call. Their value is the
complete exact source, with no generated fact fields or omitted surrounding
context. They retain the extractor's unverified usefulness judgment and a
separate `source_admission` marker (`source-episode-admission-v1`, basis
`whole_source_quote_unverified`). This does not establish truth or trust.
Wrong relevance and correction references remain possible extractor errors.
Long selected excerpts, segmented audio, structured assertions, procedures and
opinion formation keep their existing model checks. Source bytes, ownership,
current predecessor and erasure eligibility remain transaction requirements.

An explicit correction or change can reuse a supplied prior assertion's exact
subject and predicate when its current quotation refers back to that subject.
The current value must still occur in the current quotation. Structured
interpretations require independent admission review; whole-message episodes use
their distinct source admission. The stored `subject_basis_claim_id`
points directly to the original, literally grounded subject claim, including
across consecutive corrections. Commit rechecks both the current prior and that
original source; a missing or changed dependency cannot become a new assertion.
An unresolved change time remains ineligible for this exception.

Recall labels the ancestor quotation `subject_identity_only`: any old value in
it is not evidence for the current value. Both source identities accompany the
current assertion. Annotation makes an invalidated basis ineligible; erasure or
changed attribution removes dependent claims and any dependent judgment prose,
while preserving separately retained raw corrections. Preferences read the same
eligible claims. A later assertion that quotes its own full subject needs no
inherited subject dependency. Existing historical rows are not rewritten.

Review is the default admission path for new nonempty proposals. A local
installation can assign extraction and judging to the same model; older local
tier adapters use the same local tier for both requests. No separate processor
is required. Function routing honors the existing configured judging candidates
and fallback policy. An unavailable or malformed review leaves the job pending;
a valid rejection completes without that claim. The original source remains
searchable in either case. One captured outer deadline and owned job lease cover
both requests; the router also keeps each role's deadline. Review therefore adds
background work and latency, with a 1400-token output cap per batch.

Extraction has a 4096-token output allowance for its bounded proposal batch.
Review explanations can contain up to 1024 characters and are retained intact;
overlong or truncated responses remain invalid. These output bounds do not
extend the existing role deadlines, job lease or allowed attempts.

Kept records preserve the proposal's fields and add `admission_review`, containing
version `source-claim-review-v1`, a bounded reason and the actual review processor's
provenance. Its basis is `model_judgment_unverified`: a supported source assertion
is not independently verified truth, and the review reason is not a new fact.
The existing extraction provenance remains separate. Review text is excluded
from the recall assertion member projection. Historical records are unchanged.

`accepted_count` continues to mean candidates that passed the mechanical
extraction checks. `reviewed_count`, `review_kept_count`, `review_rejected_count`,
`review_response_count` and `invalid_review_count` describe the subsequent stage;
`last_review_provenance` identifies its last completed response.
`whole_source_episode_count` counts validated episodes needing no second review;
their review counts remain zero and absent review provenance remains null. `claim_count`
remains the number actually persisted. A positive extraction acceptance count
does not mean review succeeded or a claim committed. Review reasons are stored
only with kept claims, not in the job diagnostic record.

This change does not relabel or erase historical memories. Audit a retained
corpus before making content changes. Keep raw private samples on the deployment,
separate exact duplicates from distinct supporting evidence, and label model
triage as tentative. Backups, native runtime transcripts and unlinked historical
derivations have their own retention limits; canonical source erasure does not
claim to erase unsupported descendants.

Original source images also require a recoverable backup, not only retained
captions. See [source memory recovery](SOURCE-MEMORY-RECOVERY.md) for the existing
backup command's source-image coverage and its remaining recovery limits.
The claim worker skips unsupported non-text blocks before prior lookup; their
original media, stable capture metadata and independent media/vector jobs remain
retained. Supported attributed audio transcripts use their exact recognition-derived
segments for extraction and review. Completing a text job does not claim that an
image's semantics were learned or that recognized speech was independently verified.
