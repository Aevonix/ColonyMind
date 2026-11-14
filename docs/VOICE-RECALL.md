# Staged voice recall

`POST /v1/host/memory/voice-recall` is an adapter route, outside the stable public
subset. It requires the same `read` credential operation and person authority
as `/v1/host/memory/search`. It calls the same person resolver and canonical
source scoping. Carried 1.9.0 credentials use the same `memory:search` scope.
Canonical projection/annotation checks are shared. A caller never gains visibility by
naming a person, session, or utterance. There is no owner default or new sharing.
Existing search and assemble routes retain their behavior.

```json
{
  "identity": {"host_id": "voice-client"},
  "person_id": "owner",
  "session_id": "voice-session-42",
  "query": "Which tea does Morgan prefer?",
  "k": 5,
  "max_tokens": 600,
  "speculative": false,
  "utterance_id": "voice-session-42:utterance-17",
  "request_id": "voice-session-42:utterance-17:final",
  "timezone": "UTC"
}
```

`person_id`, `query`, `utterance_id`, and `request_id` are required and must not
be blank. `session_id` and `timezone` are optional. `k` is a strict integer from
3 to 5 (default 5); `max_tokens` is a strict integer from 300 to 600 (default
600). Invalid values and extra fields receive 422. `speculative` is a strict
boolean, default false. Omit the session to read person-scoped evidence only;
name it to additionally admit that session's evidence, exactly as search does.
Timezone resolution follows search, including owner contact preferences and
UTC for guests unless explicitly supplied.

The response contains `content`, `count`, `source_refs`, `watermark`,
`annotation_checks`, `retrieval`, `token_count`, `max_tokens`, `speculative`, and
both supplied IDs. `count` is at most `k`; fewer records may be returned when
there are fewer authorized hits or the evidence cannot safely fit. The hard cap
covers the complete **content string**, including the evidence warning, format
legend, JSON escaping and truncation notices. Packing uses the voice-only
`estimate_voice_tokens`: ASCII prose uses the existing estimator, rounded up
and never below its default four-character ratio; digits and hex words count
as one token per character; every non-ASCII codepoint counts as at least one,
and non-ASCII punctuation/symbols (including emoji) count as three. A stricter
configured estimate still wins. This does not change search or assemble packing.

The HTTP `token_count` remains the repository's coarse `estimate_tokens` count
for compatibility; it can be lower than the cautious count used to enforce
`max_tokens`. HTTP metadata/receipts are outside this prompt budget. A consumer
inserts only `content` into its dynamic recall prompt and retains the receipts
for freshness checks using the existing source/annotation APIs.

No serving-model tokenizer was available in the checked offline caches. New synthetic
English, digits/hex, CJK, emoji and symbol tests instead check the review's
measured GLM/old-estimator ratios: 0.82x, 1.8x, 2.45x and 3.9x respectively.
They verify both the counter and complete packed packets, including mixed
English framing and language-specific quotes, at 300/600 tokens and configured
ratios 1/4/8. This establishes the measured envelope offline; it is not an
exact-tokenizer guarantee for every possible string or a different model.

Plain unannotated quotations use JSON arrays with a shared legend:
`[source, role, reported_at, quote, optional truncation notice]`. Their complete
source/message lineage stays on the internal selected rows and source/annotation
receipts. Report time remains distinct from event time. Corrections, claim
histories, procedures, conversations, modality uncertainty, and hands provenance
retain canonical formatting and indivisible evidence handling. Overlong plain
excerpts have an explicit truncation notice; an atomic history is either complete,
an existing full-source-required notice, or omitted. Packing re-estimates each
final proposal with the cautious count, including density changes and configured overrides.
The shrink step budgets the fixed warning/legend separately from the body so
emoji-heavy quotations are not prematurely discarded despite fitting the cap.

## Partial transcript and final transcript

Start a partial with `speculative: true`. Submit the final with
`speculative: false`, the same utterance/person/session and credential, and a new
request ID. A newer request supersedes an older one for that key. An in-flight
superseded request receives **409** with `detail.code: voice_recall_superseded`
and its IDs, with no recalled content. A final also supersedes another older
final. A partial arriving after a final is refused with the same code.
Completed partials cannot be recalled from the client: once the client submits
a final, it must discard any partial result it already holds and use the final
response. `request_id` is a correlation ID; arrival order arbitrates requests.
Use a unique utterance ID for each utterance, including across reconnects.

Arbitration is process-local to one ASGI app/event loop. Route all revisions of
an utterance to the same worker (or use one worker); this is not distributed
cancellation across processes. Completed final markers are retained for 120
seconds. Completed speculative arbitration entries are removed when their request
finishes; if their caller has already left, they are removed as soon as the worker finishes.
Each authenticated **principal + credential** has independent allowances of
512 entries (active utterances plus retained finals) and 512 unfinished jobs,
including superseded or cancelled jobs. Different people/sessions under that
credential share these allowances. The value accommodates the unchanged gate's
478 distinct final requests within the retention window. A flood from one
credential cannot consume another credential's allowances or make it receive
503 `voice_recall_busy`.

The app still runs at most eight workers across credentials, so another
credential can wait for a worker under load; this is capacity isolation, not a
latency/fairness guarantee. Active jobs and unexpired finals are never evicted
for admission; exhausting either allowance for the requesting credential
returns 503 `voice_recall_busy`.

Supersession signals a cooperative cancellation flag. An already executing
SQLite read may finish, but its response is fenced and it stops before subsequent
lanes/selection where possible. A cancelled worker retains its slot until it
actually finishes; repeated partials cannot create unbounded active threads.
Client task cancellation also signals that flag. Backend errors return 503
`memory_backend_unavailable`, without source content. No packet or source evidence
is cached, so corrections and erasures are read afresh on each request. The
speculative lane retains normalized partial words and scoped index
hit ids/metadata (never evidence), described below.

## Speculative vector lane

A partial request starts `pipeline.embed_query(partial_query)` through the existing
embedding abstraction, alongside its keyword read. It returns its keyword packet
without awaiting inference. When the vector arrives, the same background job also
runs the **vector search during speech**: identity/generation checks and the
person/session scope-key prefilter before top-15, exactly as `SourceVectors.search`
does, without its embedding or hydration. The app caches only normalized partial
words, embedding identity, generation and those raw hit ids/metadata, keyed by **authenticated principal + credential + person +
session + utterance**. Nothing is persisted. A newer partial replaces the entry;
identical normalized partials with the same pipeline/store/identity share the pending
or ready vector. Entries expire after 120 seconds (pruned on cache access).
Final admission consumes the entry exactly once, before any await. A pending,
failed, expired, mismatched or missing entry means keyword-only immediately;
readiness later during that final call cannot upgrade its result. A final-only
request neither obtains the vector backend nor starts an embedding.

Reuse requires the final's case-folded Unicode word sequence to equal the partial,
or extend it as an exact prefix of at least three words and at least
`floor(0.75 × final word count)` words. Whitespace and surrounding punctuation do
not count; internal apostrophes remain part of a word (curly apostrophes normalize
to straight ones). Equal short queries are allowed. Appended negations/retractions (`no`, `not`, `never`, `except`, `instead`,
`actually`, `sorry`, and common negative contractions) reject reuse. ASR rewrites,
insertions, substantial additions, pipeline/store swaps, query-format changes or
query-transform changes all fall back. This is a conservative structural rule,
not a claim that the partial and final have identical meaning; e.g. a trailing
qualifier can still change relevance. Use a fresh utterance ID for a new request.

Each principal/credential has up to 512 speculative entries and 512 unfinished
embedding jobs. Eight separate embedding slots keep a slow provider out of the
canonical-read pool. Embed and search each have a 1.5 second timeout; embedding
timeouts leave `LIVE.running()` so background indexing can resume. A newer partial
cancels the previous task; a per-key lock prevents overlapping replacements.
At most three embeds start per utterance within the 120 second retention window,
including failures and superseded attempts. Further revisions use keywords;
identical partials still reuse a ready entry without spending another attempt.
Superseded/consumed jobs keep accounting until they exit; their late results cannot
repopulate the cache. Saturation keeps keywords.
An obsolete queued job exits before requesting inference. An already started
embedding may finish after a final, but the final never waits for it.

A ready final never embeds and never searches the index. It first builds the
keyword packet exactly as the keyword-only route does (lexical, stemmed and
claim-key lanes, canonical RRF, selection, voice packer). Only if that packet has
fewer than k records, and less than **25 ms** has elapsed since route admission
when keyword packing finishes, does it use the cached hits. Queueing, contact
lookup, keyword search and selection count toward that deadline. At or beyond
25 ms it returns the keyword packet with `local_keywords` / `unavailable`,
without hydration or a second selection. This preserves headroom for the 50 ms
final-call p95 budget; it does not interrupt an already executing SQLite read.
For an admitted fill, the generation must be unchanged,
deleted projections are excluded (one connection for the generation check,
batched deletion-log read and all canonical hydrations), hits of
messages already in the keyword packet are skipped, and then canonical `hydrate`
re-reads current source bytes under the same person/session scope until the free
slots are covered (at most `k − keyword records` hydrations). Those rows go through
the shared canonical projection, annotation and selection helpers (claims,
corrections, observed provenance, freshness re-check) limited to the free slots.
The fill does not repeat lexical media search or time-lane candidate discovery;
time filtering and named-media checks still apply. Its rows are appended after the
keyword records, and the voice packer re-fits the combined packet. If re-packing
would change or reorder any keyword record, the keyword packet is returned instead.

Sequential keyword and selection reads also share the voice worker's SQLite
connection; a time lane dispatched to another thread opens its own connection.
Packing checks empty hand provenance once per evidence version within that
packet read, avoiding repeated SQLite opens during truncation trials. Canonical
rows keep their authority fields; rendering alone uses copies after that check.
Marked hand evidence retains the shared current-state checks. No provenance
result survives into another request. Ranking, rendered bytes and token policy
are unchanged.

**Keyword-preserving fill** is a deliberate rule: a vector hit never displaces a
record the keyword-only route returns within the k/token cap. The voice route has no
calibrated cosine threshold that could say vector evidence is stronger than a keyword
hit, so it never claims that. The cost is semantic recall at k=3 when keyword noise
already fills the packet. Vector text never becomes evidence.

The final path is synchronous, cooperative SQLite/projection work only: no Lance
query, embedding, reranking or generative inference call. `retrieval.voice` is
`local_hybrid` with `retrieval.semantic: ready` when a ready cached vector was
admitted within the deadline (even if the keyword packet was already full); otherwise `local_keywords`
(`unavailable`, or `failed` for a hit validation/generation error). A ready lookup
can have no authorized hits.

The vector store's existing cosine admission floor (default 0) remains in effect.
There is no calibrated vector cutoff in this codebase. The deployment's optional
**reranker** cutoff is on a different model/score scale and cannot be applied to
cosine or RRF scores. Voice skips reranker inference and its cutoff. Offline
qualification does not establish live-search parity or calibrated abstention.

## Offline qualification

The unchanged keyword gate imports the 120 existing recall fixture records
using canonical ingestion, adds 5,000 same-person work conversations, and runs
all 78 original expected-hit queries at k=3/4/5 and voice budgets 300/600.
Its comparator has no semantic index and reranking is off. Its recall comparison
and warm serial in-process ASGI latency threshold qualify only that keyword
fixture/configuration, not live semantic retrieval or answer correctness.

The additional fixture, `sidecar/tests/fixtures/voice_recall_semantic_cases.json`,
adds 12 sources and 12 paraphrase/semantic-only queries. A separate test verifies
that every query shares no content word or Porter stem with its expected source
(after explicit function-word filtering). The quality test searches all original
sources and 5,000 distractors plus those 12 sources (5,134 sources / 10,134
messages), with authenticated in-process voice calls and frozen reference time
2026-09-05 23:00 UTC. It checks k=3/4/5 at both budgets, records expected-source
recall and keyword-noise packets, and forbids vector/reranker calls even with
reranking and a cutoff configured. It records the result without imposing a
semantic-recall threshold or pretending to measure live search.

Pre-vector baseline measured locally on 2026-10-14: semantic-only macro
expected-source recall was
**8.33% (1/12)** at every k/budget combination. Seven of the 12 queries returned
nonempty keyword-noise packets without the expected source at every setting.
The one expected hit was the rental/subletting case, via the function word
"without"; this is not evidence of semantic retrieval. These results explain
the route's keyword-only limitation even though the original keyword gate passes.

Prior keyword-only review gate: **385 passed** (87 new review tests, 270 unchanged voice tests,
1 unchanged keyword/latency gate, 27 unchanged public API tests). The first round
failed four new nonempty emoji/symbol-packet checks; accounting for the fixed
header separately corrected packing, and the second complete round passed.
Existing tests, original fixtures and thresholds were not changed.

Run the review fixes and the required unchanged gate offline with the supplied venv:

`$PYTHON` is the Python of a ColonyMind development environment and `$WORKDIR` any scratch directory.

```bash
TMPDIR=$WORKDIR \
COLONYMIND_TEST_POLL_LOOP=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VOICE_RECALL_REPORT=$WORKDIR/voice-recall-review-keywords.json \
VOICE_RECALL_SEMANTIC_REPORT=$WORKDIR/voice-recall-review-semantic.json \
$PYTHON -m pytest -c sidecar/pytest.ini \
  sidecar/tests/test_voice_recall_review.py sidecar/tests/test_voice_recall.py \
  sidecar/tests/test_voice_recall_gate.py sidecar/tests/test_public_api.py -q -s
```

The optional real-vector gate and standalone measurement runner read the external
17 MB JSON through `VOICE_RECALL_EMBEDDINGS`; both skip if it is absent. It is never
copied into the repository. The gate exercises authenticated partial/final HTTP,
a fresh temporary Lance index, all 12 semantic queries and all 78 original keyword
queries at every k/budget, and reports keyword-only, q75-prefix and full-query
macro expected-source recall. The replay embedder returns the recorded **real**
Qwen/Qwen3-Embedding-8B 4096-dimensional vectors after a 70.1 ms delay, completed
before timing the final POST. Full-query vectors are an upper-bound comparison
staged through a speculative POST too; the final never embeds its text.

The supplied single distractor vector is repeated for 10,000 background messages;
these are not 10,000 independently embedded texts. Canonical sources/messages are
still genuinely ingested and hydrated. This qualifies a seeded ~10k-message
store, not varied live vector neighborhoods or concurrent traffic. The report
states exact source/message/chunk counts. Warm serial final-call p50/p95 (nearest
rank) use 90 queries per lane at k=5/600. A separate late lane holds every embedding
pending beyond 75 ms while its final returns keyword-only, then releases that job.
The local qualification budget is **50 ms p95**. Reports include lane readiness,
per-query receipts and all gate decisions; no report silently counts a vector
lookup timeout as semantic success.

```bash
TMPDIR=$WORKDIR \
VOICE_RECALL_EMBEDDINGS=$WORKDIR/voice-recall-embeddings.json \
VOICE_RECALL_VECTOR_REPORT=$WORKDIR/voice-vector-results.json \
$PYTHON \
  benchmarks/source_recall/measure_voice_speculation.py
```

Staged results are summarized below; detailed measurement
receipts are written to the external `VOICE_RECALL_VECTOR_REPORT` path.


### Initial speculative recall measurement

The first staged candidate failed two gates (q75 lost keyword receipts at small
caps; ready final p50/p95 60.5/101.7 ms). Profiling one final call: ~30 ms was the
synchronous Lance brute-force cosine search over 10,135 4096-d rows plus per-row
eligibility reads, ~15 ms was hydrating 15 hits (two SQLite connections each), and
the rest was selection over the larger fused set. RRF one-vote-per-modality with
the selector's lexical/hybrid alternation let vector rows displace keyword rows.
Revision: search during speech; final hydrates only free slots; keyword-preserving fill.

Expected-source macro recall, keyword-only → q75 / full (semantic 12, keyword 78):

| k/tokens | Semantic | Keyword |
|---|---|---|
| 3/300 | 8.33% → 66.67% / 66.67% | 92.31% → 92.31% / 92.31% |
| 3/600 | 8.33% → 58.33% / 58.33% | 92.31% → 92.31% / 92.31% |
| 4/300 | 8.33% → 66.67% / 66.67% | 96.15% → 96.15% / 96.15% |
| 4/600 | 8.33% → 75.00% / 75.00% | 96.15% → 96.15% / 96.15% |
| 5/300 | 8.33% → 83.33% / 83.33% | 96.15% → 96.15% / 96.15% |
| 5/600 | 8.33% → 91.67% / 91.67% | 96.15% → 96.15% / 96.15% |

Every keyword-only hit is retained per query in every setting (0 displaced).
Warm final-call p50/p95 ms (90 samples, k=5/600, same 10k store): keyword-only
17.40/26.77, q75-ready 20.29/46.93, full-ready 20.51/48.93, late 18.03/26.74.
The ready tail is the keyword route's own selection on four noisy semantic queries
(~45–50 ms keyword-only) plus ~5–7 ms of fill; p95 margin to 50 ms is small.
All five gates pass. Receipts: `$WORKDIR/voice-vector-fix.json`.
Nothing live was activated or modified; the embedding file stays outside git.

### T2 repair qualification (2026-10-14)

The first repair round preserved keyword recall but lost semantic hits when noisy
keyword queries exceeded the new 25 ms fill deadline. Profiling found repeated
hand-provenance SQLite reads during packing. Packet-local empty-provenance checks
and a shared worker connection restored headroom without changing packed output.
The second round passes all eight assertions in the single qualification gate.
Existing test files and both query fixtures remain unchanged.

Expected-source macro recall, keyword-only → q75 / full (semantic 12, keyword 78):

| k/tokens | Semantic | Keyword |
|---|---|---|
| 3/300 | 8.33% → 66.67% / 66.67% | 92.31% → 92.31% / 92.31% |
| 3/600 | 8.33% → 58.33% / 58.33% | 92.31% → 92.31% / 92.31% |
| 4/300 | 8.33% → 66.67% / 66.67% | 96.15% → 96.15% / 96.15% |
| 4/600 | 8.33% → 75.00% / 75.00% | 96.15% → 96.15% / 96.15% |
| 5/300 | 8.33% → 83.33% / 83.33% | 96.15% → 96.15% / 96.15% |
| 5/600 | 8.33% → 91.67% / 91.67% | 96.15% → 96.15% / 96.15% |

All reviewed tables match. Every keyword receipt is preserved per query at every
setting, including the semantic queries' keyword noise. Counts remain 5,134
person-scoped sources, 10,134 messages and 10,135 indexed chunks (one other-person
fixture is indexed but excluded by scope).

| Final lane, k=5/600; 90 samples each | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|
| Keyword only | 11.02 | 16.79 | 21.78 |
| q75 ready | 14.69 | 21.37 | 27.56 |
| Full ready | 14.50 | 21.63 | 26.82 |
| Late, keyword fallback | 12.52 | 18.72 | 23.88 |
| Concurrent embed, keyword fallback | 13.01 | 18.16 | 23.27 |
| Concurrent search, keyword fallback | 33.22 | 37.77 | 43.20 |

For each concurrent sample, the previous partial is paused inside an actual SQLite
statement using a progress callback. The final starts while that read and the
recorded 70.1 ms embedding, or a real Lance search, are still in flight. Search has
no delay or substituted hits. Finals sample readiness once and return keywords;
the partial receives 409 and its worker finishes after the final. Both concurrent
fallback lanes retain the exact baseline expected-source recall. This qualifies
the controlled local overlap, not unpaused production traffic or live hardware.
All p95 values are below 50 ms. q75-ready metadata reports 84/90 admissions;
full-ready reports 90/90. Deadline fallbacks are counted as keyword-only.

The original harness did send the full query; its 14 text collisions were among
prefixes. Replay now selects an explicit case/variant and audits the actual vector
passed to Lance against its recorded SHA-256. All 1,175 searches match: 545 full
and 630 prefix searches. All 12 semantic full/prefix vectors and raw hit rankings
differ at k=5/600, despite equal final recall totals. The full-query numbers above
are therefore independently verified.

Validation: 611 regression tests passed, 20 skipped; the real-corpus/concurrent
gate also passed. The 17 new tests comprise 16 lifecycle/deadline/packing checks
plus that gate. Detailed receipts:
`$WORKDIR/voice-t2-round2.json`.
