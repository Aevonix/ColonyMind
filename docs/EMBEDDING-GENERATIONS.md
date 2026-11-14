# Embedding generations

Canonical source messages and graph facts are retained independently of their
embeddings. A vector is a replaceable search projection. Equal vector dimensions
do not establish that two models produce compatible embeddings.

The runtime binds its vector store to a document identity: requested model alias,
reported serving model, declared weights or deployment revision, dimension,
document format, normalization, quantization and any explicit requested dimensions.
`EmbeddingIdentity.document_identity` exposes that value. Query format and the
client query transform are separate query settings.
The historical quantization identity field remains `unspecified` for current
pipelines: these providers do not apply a configured quantization option.
`COLONYMIND_EMBED_QUANTIZATION`, `COLONYMIND_EMBED_CACHE_SIZE` and
`COLONYMIND_EMBED_BATCH_WINDOW_MS` have been removed because they did not control
runtime behavior. Remove them from the process environment and the configuration's
`environment` mapping to upgrade; startup names any retired variable still set.
Configure quantization at the serving provider and declare its weights revision,
then rebuild after changing served weights. Historical catalog identities remain
readable and still require a matching document identity for reuse.
The requested alias and serving model are separate fields. Missing revision or
serving identity is recorded as `unknown`. `COLONYMIND_EMBED_REVISION` is an operator
declaration, not verified weights attestation. An endpoint that silently changes
weights behind the same alias without reporting a change cannot be detected
from that alias alone; declare a new revision and rebuild after such a change.
Endpoint addresses and credentials are not index identity and are not copied
into index metadata.

The catalog retains its historical fingerprint tokens and identity JSON. When
reading an existing generation it verifies that token, then compares document
identities. Upgrading with the identity query transform neither rebuilds the index
nor rewrites its active pointer. Query-only changes reuse the generation; a change
to any document-producing field still requires a rebuild. Unverified legacy
collections remain unverified.

The query transform is `identity` or `query-adapter-1:<sha256>`. The owner recall
path supplies its pinned revision in calibration metadata and keys its transformed
query memo by it; document vectors and the active index generation stay unchanged.
The `plasticity.query_adapter` portable global and shown revisions each contain
float32 `U.npy`, `V.npy` and `revision.json`. Each factor digest is SHA-256 over U's
then V's bytes in C order; a zero residual has the literal digest `identity`. With
identity shown factors the transform uses the global digest. Otherwise it uses
SHA-256 of the compact JSON array of global and shown digests, so either factor
scope invalidates calibration. The registered driver binds both scopes and the
refitted cutoff to one serving closure. Explicit identity preserves the old cutoff
fingerprint. Contacts, guests, lessons, claim pairing and novelty retain the base
vector; only authenticated owner recall applies the learned transform.

Before release, run `test_embedding_identity_split.py` and
`test_recall_fingerprint_adapter.py` against a clone. Check that the index's active
generation and the configured cutoff stamp are unchanged for `identity`. Compare
the actual model, served revision, formats and query instruction with the lesson
bounds entry, and record matching bounds before relying on a different identity.

The pipeline retains its query instruction for its lifetime, and source semantic
recall uses that same query formatter. The API embedder checks response cardinality,
ordering, finite values and dimensions before returning vectors. A changed
reported serving model cannot silently reuse the old pipeline. This change does
not add dynamic embedding provider configuration; selecting a different
embedding provider still uses the existing deployment configuration path.
Index promotion itself takes effect in the running store without a restart.

For the `openai_api` text provider, `COLONYMIND_EMBED_DIMS` specifies the expected
output width; it does not request dimension reduction. To explicitly ask a
supporting endpoint for a width, set `COLONYMIND_EMBED_REQUEST_DIMS` to the same
positive integer (or set `EmbeddingConfig.request_dimensions`). With this option
unset, requests continue to omit `dimensions`. The client never truncates
vectors, and rejects responses with a different width. This option is not
supported by local or multimodal providers. A multimodal startup that selects it
falls back to text-only operation through the existing initialization path.
Changing the explicit request option changes the embedding identity even when
the expected output width is unchanged; rebuild before semantic use. Unchanged
configurations retain their existing generation identity.

Existing Lance collections have no trustworthy embedding identity. They remain
on disk as an unverified legacy generation, and semantic search through the
managed runtime refuses to compare them with a newly configured model. Canonical
lexical source recall remains available. The
installer's optional vector dependency remains optional.

The existing `colonymind migrate-tier` and `/v1/host/memory/migrate` operation rebuild
all retained text collections into a separate Lance directory. The server
registers the same store semantic recall reads with that operation. Every
retained text collection is rebuilt from its retained text and metadata. A model-filtered partial rebuild cannot be promoted because
it would leave the replacement incomplete.

The generation catalog, active pointer and exact-ID deletion fences use the
existing source ledger, not another database service. Rebuilding is resumable:
already staged IDs, including new live writes, are retained. A failed batch
leaves the prior active pointer intact. A complete generation is promoted by
one SQLite transaction. Old generations remain available for inspection or a
separately qualified recovery. This packet provides no unattended old-generation
rollback command and does not restore authoritative memory or authority state.
The ordinary migration API rejects concurrent rebuilds in one server process.
The CLI stops on interrupted/resumable or unavailable status and explains how
to resume without claiming completion. `--wait-seconds` bounds waiting (600
seconds by default); reaching that limit does not cancel the server job.

Managed vector maintenance can tombstone and remove projection ids from active, staged and retained
generations. Source forgetting uses the curated memory journal instead: historical vectors and native rows
stay intact, and the shared source fence suppresses their evidence in recall. A forget response contains the
curated receipt and no physical-cleanup status. Unlinked historical vectors cannot acquire invented source
ownership from matching text. Their source provenance remains unknown.

Compaction is also routine, because every append writes a manifest that lists
every fragment of its table: a table nothing compacts grows its version history
quadratically (one upgraded store held over forty times as many bytes of
manifests as of data). One background task per store, never a request, compacts one table at a
time and deletes its older versions:

- once per night crossed (the consolidation's boundary), scheduled by the mind's
  tick whether or not the mind is on: every table with an older version;
- every five minutes, a table holding `COLONYMIND_VECTOR_COMPACT_VERSIONS` (1000)
  versions or more.

A pass reads every retained manifest, about 130 MB/s whatever it prunes, so a
table whose manifests exceed `COLONYMIND_VECTOR_COMPACT_DAY_BYTES` (1 GiB) is
not compacted by the threshold: it waits for the nightly pass, and for at most
a day. Between two tables the task pauses as long as the previous one took (at
most 30 s). Each table's pass is logged at info with its reason, versions and
size before and after, and its duration; a failure is logged as a warning and
the pass goes on.

A table's pass holds the store's write lock from start to end, so writes to the
store (source vector updates and projection maintenance) wait for it, while reads
go on. Tables created by earlier releases carry Lance's own auto-cleanup
(`lance.auto_cleanup.interval` 20 and `older_than` 14 days in the table's
config; the lancedb 0.39 Python API can neither read nor unset it), which runs
inside a commit: a commit beside a first pass once deleted the manifests that
pass was pruning, and the pass failed after 13 minutes. Under the lock no
commit, and so no auto-cleanup, runs beside a pass. A read opens the latest version and reads
its files as it goes, so a pass prunes nothing a read may still need: it waits
for the reads begun before it (at most `READ_DRAIN_SECONDS`, 300 s, then it
logs how many still run and prunes), compacts, prunes only what is older than
the version current when it began, waits for the reads begun before its
compaction, and then prunes all but the latest version. A version manifest that
vanishes under a prune anyway (a cleaner in another process) is retried once.

Rebuilding image embeddings requires a qualified model that can reproduce that
image embedding space. The text rebuild refuses retained image-vector rows
instead of replacing them with caption vectors and claiming compatibility.
Canonical retained originals and their fallible descriptions remain separate.
[Semantic source and caption candidates](SOURCE-SEMANTIC-RECALL.md) use their own
scoped text projection in the managed generation, followed by the existing
single context selector and budget. Caption retrieval quality still requires
measurement with the selected embedding and reranking models.

`test_embedding_generations.py` uses real LanceDB for identity mismatch, changed
dimensions, non-destructive failed update, failed rebuild/resume, new arrivals,
erasure, and the real migration/forget API. Controlled vectors test distinct
spaces; a separate private neutral qualification uses the actual LAN embedder.
That small qualification does not establish comparative retrieval quality or
the performance of re-embedding a production corpus.
