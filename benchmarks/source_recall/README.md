# Source-recall comparison

This harness compares lexical and canonical hybrid recall through the production
collection and selection paths. The earlier three-arm reference results are
retired. Run the current harness to produce results for a declared configuration;
no measured quality claim is supplied by the retired reference.

## What runs

Both arms run the production recall path, `colonymind.memory.search.collect_sources`
then `select_memory`, over the actual canonical SQLite source ledger, source-claim
projection, LanceDB vector store and shared `RecallSelector`, and they measure the
packet a host's turn receives (`TURN_PACKET` in `run.py`, recorded as `packet` in the
output): behind a verified cutoff whose rerank returned it has no count limit, so
every candidate that passed the cutoff is placed in rank order within 6,000
characters; with the cutoff unverified or the rerank missed it holds at most five.
An explicit memory search names its own limit; a turn does not, and neither does
this harness. There is no host allowance beside the packet here: the runner in
`benchmarks/public_memory` reads the whole assembled context through
`/v1/host/context/assemble` for that. OpenAI-compatible extraction and embedding calls
and the Cohere/Jina-style `/v1/rerank` transport use operator-supplied endpoints.
Model calls run sequentially. Extraction is attempted once per inserted source;
failures remain reported in the job status rather than silently retried to pass.

| Arm | Candidate inputs |
| --- | --- |
| `lexical_only` | The same collection with semantic recall unavailable: canonical source FTS, up to 10 hits |
| `canonical_hybrid` | Canonical source FTS plus canonical source semantic search, as production collects them |

Extraction runs as an install's claim worker runs it: each source is extracted after the sources before it have
been embedded, so the prior records offered to the extractor come from both lanes, the keyword search of the new
message and the source vector search with it (claims v18). `--no-priors-by-meaning` is that lane's ablation, made
with the claim projection's own switch as an install makes it (`COLONYMIND_PRIORS_BY_MEANING=0`, which
`mind.faculties.priors_by_meaning: false` exports): the priors come from the keyword lane alone, as they did before,
and nothing else changes. Which lanes ran is part of the extraction declaration
(a state directory extracted one way is never resumed the other) and of the output (`priors_by_meaning`).
Captured extraction state from before this change was formed with the keyword lane alone.

Corpus records all belong to a synthetic owner; the six guest cases test abstention for private information. The fixture's
team/public annotations do not prove shared-authority behavior. Synthetic model
generation labels are not an actual model-swap test; real generation mismatch
and erasure races are covered separately by `test_embedding_generations.py` and
`test_source_vectors.py`.

## Reproduce

Run from the repository root in a Python environment with ColonyMind's dependencies
installed, including LanceDB, PyArrow and Pillow. No additional memory framework
or database service is installed by the harness. Use disposable directories.
Never run two harness processes against the same state directory.

Set these environment variables to explicit benchmark model endpoints and names:

| Required | Meaning |
| --- | --- |
| `COLONYMIND_BENCH_CHAT_BASE_URL`, `COLONYMIND_BENCH_CHAT_MODEL` | Local OpenAI chat endpoint ending in `/v1`, extraction model |
| `COLONYMIND_BENCH_EMBED_BASE_URL`, `COLONYMIND_BENCH_EMBED_MODEL`, `COLONYMIND_BENCH_EMBED_DIMS` | Embedding endpoint, model, full dimensions |
| `COLONYMIND_BENCH_RERANKER_BASE_URL`, `COLONYMIND_BENCH_RERANKER_MODEL` | Rerank endpoint base **without** `/v1`, model |

Optional keys are `COLONYMIND_BENCH_CHAT_API_KEY`, `COLONYMIND_BENCH_EMBED_API_KEY` and
`COLONYMIND_BENCH_RERANKER_API_KEY`. Optional `COLONYMIND_BENCH_LOCAL_HOSTS` declares
comma-separated LAN hostnames for extraction; loopback and private IP ranges
already follow the runtime router's local policy. There is no cloud fallback.
`COLONYMIND_BENCH_RERANKER_PROMPT_STYLE=qwen3` selects the Qwen3 template.
`COLONYMIND_BENCH_CHAT_WEIGHT_REVISION` records a known revision, otherwise unknown.
`COLONYMIND_BENCH_EMBED_QUERY_INSTRUCTION` overrides the query prefix; its default is
`Instruct: Given a search query, retrieve relevant memories that answer it\nQuery: `
with an actual newline. Do not use shell tracing when setting API keys.

```sh
python benchmarks/source_recall/run.py --state-dir /tmp/recall-comparison \
  --output /tmp/recall-before.json --threshold 0.95 --ranking-format verbose-claim-json
python benchmarks/source_recall/run.py --state-dir /tmp/recall-comparison \
  --output /tmp/recall-after.json --threshold 0.95 --ranking-format grounded-quotation-bundles-v1
python -m pytest benchmarks/source_recall/test_harness.py -q
```

The first command extracts the neutral corpus; the second reuses that exact
prepared state. The fixture, extraction declarations and observed embedding
identity are checked before reuse. The harness refuses an existing unmarked
state directory, never loads deployment credentials, and omits endpoints and
API keys from result artifacts. Its transport smoke test uses a controlled local
HTTP model and real SQLite/Lance to check transport and resume behavior; focused
tests also check assessment and capture replay. These are not semantic-quality
measurements.

The frozen fixture SHA-256 is
`00d30ffd6829013445c766ba2244d55142247e99f7b4e68714c1c022ea8dcdff`.
All names, places, source text and example access phrases are synthetic.
Local run outputs contain the measured per-query assessments. Use a new
directory for another model or fixture. These previously inspected holdout questions are now a regression set;
a future claim of generalization needs a fresh held-out set.

## Reusing selection observations offline

New local output files include a `replay` object on each query/arm and caption
result. It records the full preselection beliefs and quotations, their order,
selection parameters, the four selector environment values, safe calibration
metadata and its original fingerprint, actual reranker requests and returned
index/score rows, and the selected typed rows. `selection_sources` identifies the
benchmark, selector and packer source bytes by SHA-256. The existing rendered
context and assessment remain beside the capture.

The capture contains no configured endpoint, key, arbitrary environment or
exception body. A provider failure records only its exception class; a cancelled
provider records `cancelled`, which alone does not distinguish the selector's
timeout from other cancellation. A returned result is not automatically valid:
the selected rows retain the selector's scored/unavailable outcome. These files
contain the supplied source text and belong in the operator's local results,
not automatically in a public reference file.

An offline consumer restores the recorded inputs and parameters, injects the
recorded scores into the actual `RecallSelector`, and compares selected rows,
rendered context and `assessment.assess` with the retained result. It must verify
the exact query, ordered document text and `top_k` for every request **after**
selection. An assertion raised inside the callback can be caught by the
selector's normal fail-open path and is insufficient to reject an invalid
replay. `replay_observation` in `test_harness.py` demonstrates this check without
calling a model. It also demonstrates reproducing the recorded calibration
equality decision using safe metadata after omitting the endpoint, rather than
claiming that the original endpoint can be reconstructed.

A turn's packet sends the reranker at most 20 documents, a bound of its own that
the packet's count does not change; an explicit limit instead submits four times
that limit (five, eight and twelve records submit up to 20, 32 and 48 documents).
A capture of 20 scores cannot qualify a request for 48. Reject changed or
additional documents rather than assigning missing scores or silently using
fallback. Applying only the packer to a fixed ranked list can compare character
use at different budgets, but is a packing-only experiment. A wider selector
comparison needs the additional actual observations and explicit treatment of
any changed batching. Output from before the turn packet (a five-record packet,
with no `packet` field) is not comparable row for row: a packet can now hold more
than five sources. The 6,000-character budget is unchanged. Existing expected-source labels measure
coverage, validity, forbidden evidence, abstention and conflicts; extra eligible
sources need separate relevance assessment before they can be called junk.


## Source-only evidence quality cohorts

`--source-only` reuses the same source ledger, LanceDB index, embedding provider,
annotation expansion and `SelectionCapture` for one `canonical_hybrid` arm.
It merges ten lexical and fifteen semantic hits before the existing twenty-document
reranker bound and the turn packet within 6,000 characters. It skips extraction and
caption extras. No production default changes. This boundary tests
retrieval of imported evidence, not native memory formation or final answer quality.

In this mode each fixture record may set `role` to `user`, `assistant` or `tool`
(default `user`). Its `content` string is stored exactly, including raw tool JSON;
tools are not recast as human statements. Optional fixture `annotations` contain
`id`, `target`, `excerpt`, `correction` and `author_principal`. These go through the
real source-annotation API and preserve original revision and annotation ancestry.
All sources belong to one synthetic owner; this is not an authority test.

Optional per-query `relevance` maps source IDs to `answer_useful`, `context_only`
or `irrelevant`. `required_evidence` lists exact necessary spans. These labels
are consumed only after selection. `useful_packet_pass` requires the existing
strict result, required evidence and no irrelevant or unlabeled selected source.
`source_utility_pass` permits explicitly labeled `context_only` records when a
question has no known answer. A requested inspection or partial transfer can
explain missing evidence; neither establishes a completed result. The stricter
abstention score still requires an empty packet. Both scores are reported, and
neither grades a generated answer. Freeze these labels before measuring a case.
Annotation bundles are scored against their corrected original source; exact
annotation IDs and versions remain in the replay artifact. Eligible extra sources
are therefore no longer silently called useful. A complete packet can still be
misinterpreted by a model.

Use `--split development` to measure only calibration questions, freeze one
candidate decision, then use `--split holdout` with the same marked state.
New held-out questions and labels must be authored before viewing their scores;
previously inspected questions are regression cases. Keep private evidence and
locally authored held-out fixtures outside the public repository.

```sh
python benchmarks/source_recall/run.py --source-only --split development   --fixture /private/frozen-evidence.json --state-dir /tmp/source-quality   --output /private/development-first.json --threshold "$TRIAL_CUTOFF"
python benchmarks/source_recall/run.py --source-only --split holdout   --fixture /private/frozen-evidence.json --state-dir /tmp/source-quality   --output /private/holdout-first.json --threshold "$TRIAL_CUTOFF"
```

Only the explicit benchmark embedding and reranker environment variables are
required here. `COLONYMIND_BENCH_RERANKER_REVISION` and
`COLONYMIND_BENCH_RECALL_INDEX_GENERATION` optionally declare known revisions;
otherwise they remain `unverified`. The calibration metadata uses the serving
correction-first candidate format and records embedding dimensions. The explicit
trial cutoff and matching stamp are not a new qualification of that cutoff.
Declare the trial cutoff before examining held-out results. A new query
instruction or embedding identity requires a new disposable state. A reranker
change can reuse the same prepared corpus, with new scores and calibration metadata. Preserve
first failures, exact candidate order, returned scores and selected evidence.
Changing cutoff can reuse a captured score set only if the actual outgoing query,
ordered documents and top-k remain identical. Changing representation or expanding
the candidate cap requires new measurements.
