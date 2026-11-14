# Function routing on the existing model pool

For shared capacity limits across native Hermes and supporting model calls, see
[Shared inference capacity](INFERENCE-POOL.md). Function-role selection and
request admission are separate: role metadata alone does not reserve capacity.

ColonyMind's shared `LLMRouter` now selects a named function and tries only that
function's eligible local candidates. It keeps its object identity when host
configuration changes, so retained extractors, thinkers, planners and workers
see updates. An active request retains its old model, endpoint, credentials,
capabilities and fallback order until that request finishes. A later request
uses the new configuration.

Successful responses include `prior_attempts`: earlier binding/model names,
`failed` versus `skipped` status, and exception class, `RequestBudgetExceeded`,
`missing_final_answer`, `incomplete_final_answer` or cooling-down reason.
Endpoint addresses, credentials and exception messages are omitted. Ineligible
candidates are not attempted and do not appear as failures. The returned model
and binding remain the actual processor; returning a fallback alone does not
prove that every earlier configured candidate was attempted.

This packet covers OpenAI-compatible chat completions for `chat`, `reasoning`,
`planning`, `extraction`, `judging`, `vision` and `coding`. It does not replace
Hermes's primary conversation routing, voice/STT/TTS transports, embeddings or
reranking. Those remain separately configured consumers. There is no new
service, fleet scanner, workflow engine, cloud authorization path or automatic
model benchmark.

## Configuration

The existing state-directory `.colonymind-llm-config.json` remains authoritative.
Existing `provider`, `baseUrl`, `apiKey` and `models.small/medium/large` bindings
work for supported local OpenAI-compatible endpoints. Object specs may declare
`supportsTools`, `supportsVision`, `supportsJsonSchema`, `contextTokens`, `latencyMs`,
`tokensPerSecond`, `concurrency`, `weightRevision` and `reasoningTokens` (below). Existing
`usefulContextTokens` remains a compatible context hint.

The optional `modelPool` provides more than three named bindings, independently
of machines or parameter count. `functionRoles` chooses ordered candidates and
requirements. Names express deployment roles, not hardware identifiers.

Optional `taskRoles` maps a supported task name to an existing function role.
`source_claim_review` defaults to `judging` and can be assigned independently of
other judgment tasks. Mapping it to `extraction` uses that role's candidates and
deadline for memory admission review, including the worker's outer time bound.
The claim records the role and binding actually used. Changing this mapping does
not change the global judging or planning roles.
For example, `"taskRoles": {"source_claim_extraction": "reasoning"}` sends source
memory formation through reasoning while ToM and appraisals retain
their existing selection. Capability hints, outer extraction budgets and actual
dispatch use the same selector. An explicit per-call `function_role` takes
priority. Omission or an empty map preserves existing behavior; unknown tasks or
roles reject the configuration. Supported task names are declared in
`colonymind/router/functions.py` as `TASK_ROLES`. Routing status reports
the overrides as `task_roles`; each active request keeps its selected snapshot.

Appraisal formation uses `source_appraisal`, which defaults to extraction. When
the existing source worker must compare a retained appraisal, integrate an owner
correction or resolve an open incident, it selects `source_appraisal_revision`,
which defaults to reasoning. `self_judgment`, the opinion pass that proposes the
agent's stances, also defaults to reasoning. All three can be assigned
independently through `taskRoles`; each selected function supplies the background
request budget. Ordinary recollection does not wait for these background updates.
Assigning a role does not enable a disabled operator; the opinion pass runs only
while `mind.faculties.opinions` is on. The recorded processor metadata names the
task, model and configuration that actually produced each new appraisal or stance.

Working-memory branch reads use the `consult` task, which defaults to `reasoning`.
Set `"taskRoles": {"consult": "reasoning"}` to select the deployment's reasoning
role explicitly, or map it to another existing function role. Both the branch
answer and its sampled summary comparator use that task with no tools. Its
canonical material prefix stays byte-identical while the retained branch spans
stay unchanged; endpoint-reported cached tokens determine the measured cached
share. The working-memory cap and hourly budget apply in addition to the selected
role's context and deadline limits. See [Working memory](WORKING-MEMORY.md).

```json
{
  "provider": "local",
  "apiKey": "local-no-key",
  "models": {},
  "modelPool": {
    "interactive": {
      "model": "deployed-fast-model",
      "baseUrl": "http://192.0.2.1:8000/v1",
      "supportsTools": true,
      "contextTokens": 64000,
      "maxTokens": 4096,
      "latencyMs": 250,
      "tokensPerSecond": 80,
      "concurrency": 4
    },
    "deliberate": {
      "model": "deployed-reasoning-model",
      "baseUrl": "http://192.0.2.1:8000/v1",
      "supportsTools": true,
      "contextTokens": 128000,
      "maxTokens": 8192,
      "latencyMs": 2000
    },
    "visual": {
      "model": "deployed-vision-model",
      "baseUrl": "http://192.0.2.1:8000/v1",
      "supportsVision": true,
      "contextTokens": 64000
    }
  },
  "functionRoles": {
    "chat": {"candidates": ["interactive"], "maxLatencyMs": 500},
    "reasoning": ["deliberate", "interactive"],
    "planning": ["deliberate", "interactive"],
    "extraction": ["interactive", "deliberate"],
    "judging": ["deliberate"],
    "vision": ["visual"],
    "coding": ["deliberate", "interactive"]
  }
}
```

Every public example is neutral. Real credentials belong only in private host
configuration. `modelPool` supports at most 64 bindings and each role at most
eight candidates. A candidate spec may override the inherited `apiKey`,
`baseUrl`, `maxTokens` and `extraBody`. A `tier` is optional legacy metadata;
the selected function is reported separately.

A role object accepts `minContextTokens`, `maxLatencyMs`, `minTokensPerSecond`,
`minConcurrency`, `timeoutSeconds` and `deadlineSeconds`. Capabilities and
performance figures are deployment declarations, not router measurements.
Unknown values do not satisfy explicit constraints. Selection uses declared
order after filtering, without guessing intelligence from a model name.
Concurrency describes advertised capacity; this packet does not reserve slots
or measure current contention. Independent registration can update this same
configuration later.

Without explicit function mappings, chat uses SMALL; extraction uses SMALL then
MEDIUM; reasoning/planning use LARGE, MEDIUM, SMALL; judging/coding use LARGE
then MEDIUM; vision uses only explicitly configured VISION. Missing model
bindings are omitted. Provider presets and discovered file size do not invent
available function candidates.

Explicit pool candidates need `supportsTools: true` for tool-bearing requests.
Existing tier bindings preserve unknown tool capability for compatibility;
status marks these `legacy_unknown_tools_allowed: true`. After a real successful
tool exercise, the deployment can replace the unknown with an explicit
capability. There is no unknown-capability exception for vision.

After verifying a serving endpoint's strict JSON-schema support, declare
`supportsJsonSchema: true` on that binding. Memory formation then supplies its
output schema with the individual request. Other calls keep their normal output
format. This declaration defaults to false and is not inferred from the model's
name or an HTTP success alone. The schema must actually constrain the output,
including arrays and the object alternatives used by the memory consumers.

## Reasoning allowance

A caller's `max_output_tokens` is the budget of its answer. A model whose thinking
is on spends output tokens reasoning before that answer, so a request capped at
the answer's budget alone can end (`finish_reason: length`) before any answer is
written. A binding whose own `extraBody` turns thinking on is therefore sent the
caller's answer budget plus a reasoning allowance, 4096 tokens by default. Exactly
two switches are read, both under `extraBody.chat_template_kwargs` and both only
when literally `true`: `enable_thinking` (the vLLM and SGLang chat templates) and
`thinking` (templates that name it so). Nothing else is detected: not the model's
name, a server-side default, a switch in the system prompt or any other field.

`reasoningTokens` on a binding spec sets the allowance: a number of tokens from 0
to 1048576, read as leniently as the binding's other numbers (`4096.0` is 4096,
and a fraction is cut as `maxTokens`' would be), since an invalid persisted
configuration drops all routing at the next start. Set it for a model that always
thinks, or thinks by a server default, with no switch in its request body: without
it such a binding gets no allowance. `0` sends exactly the caller's budget.
Thinking itself is never turned off by the router.

The allowance never takes the request past the binding's `maxTokens`, or past its
declared `contextTokens` after the prompt's room: the router's estimate of the
messages, the schema where it is sent and any tool schemas, times 1.25 for what
that rough estimate misses (the chat template, formatting, images). Eligibility
still needs room only for the answer itself; when those limits leave less, the
allowance shrinks and the answer keeps its whole budget. If the endpoint still
refuses the larger request as too long for its context window, the same binding is
asked once more with the answer's budget alone, inside the same attempt's time,
before the attempt fails; that refusal never cools the endpoint down. A caller
that sends no `max_output_tokens` still gets the binding's own `maxTokens`. An
answer that is still truncated remains an `incomplete_final_answer` attempt, as
before.

A completion that ran out while thinking still used its whole cap. The router sums
what the endpoints reported for a call's failed attempts: on an answer as
`prior_usage`, and on the `RuntimeError` it raises when no attempt completed as
`usage`. `colonymind.usage.function_usage`/`charged_tokens` read either, and every
caller that charges a token budget (the night's consolidation and lesson call, the
typed decisions and plain summaries the mind charges to the day, deliberation,
composition, the identity check, a claim dry run's meter) charges a failed call's
reported tokens as it charges an answered one's.

`GET /v1/host/models` reports each binding's `reasoning_tokens`, each
qualification observation records the answering binding's as
`binding_reasoning_tokens` beside `client_max_tokens`, and
`LLMRouter.function_output_tokens(context=...)` returns the largest request cap among the
role's eligible bindings (only the first with `allow_fallback=False`), for callers that size their own
token budgets (the night's consolidation does). The field is optional: a
configuration without it loads unchanged, and an invalid value rejects the whole
configuration like any other invalid field.

Every caller on a thinking binding gets the allowance, not only the night's: the
owner-turn instruction read, the gate's judge, composition, typed decisions and
the rest. Such a call may generate more and take longer, always within the
deadline it already had: the role's `timeoutSeconds`/`deadlineSeconds` and the
caller's own bound (the owner's turn waits on the instruction read at most its
10-second `TURN_DEADLINE` plus a second's grace, and keeps the reader's own
conservative answer when the read is not in time).

A prompt-only fallback receives the original task without an unsupported schema;
its context budget excludes that unsent schema. Every response still passes the
consumer's source, field, scope and revision checks. Structured output improves
format reliability, not the truth or usefulness of a memory. A server rejecting
a declared schema with a nonretryable HTTP error remains an explicit failed
attempt; the router does not silently strip the contract and retry that endpoint.

Only declared OpenAI-compatible transport is supported by this function path.
An existing Ollama deployment must explicitly declare `protocol: "openai-chat"`
and provide its compatible `/v1` URL, or configure that compatible endpoint as
`provider: "local"`. Native Ollama or other provider-specific protocols are
rejected during configuration; they are not accepted as unusable candidates.
The older direct-construction tier router API remains for its existing callers.
No ColonyMind consumer currently requests router streaming. `stream=True` is
rejected before a function call, and Hermes retains its own streaming behavior.

## Reload and fallback

Function routing validates final text before accepting a completion. Empty,
reasoning-only and truncated responses can use the next eligible candidate
within the same request budget. A requested tool turn remains valid; it does
not count as final text for a consumer that needs an answer to persist. Failed
output validation does not put the endpoint into a shared cooldown. This adds
neither a retry of the same candidate nor another time allowance. The direct
legacy tier API retains its existing compatibility behavior.

Calls and metadata reads check the configuration file's inode, size and mtime.
A changed file is parsed and validated into a separate snapshot, then published
atomically. Invalid or partial edits retain the last valid snapshot and expose
`reload_error`; an invalid first configuration leaves inference unavailable,
with no implicit provider default. Configuration files should be replaced
atomically. `POST /v1/host/configure` validates, writes a private file atomically,
and updates the same router object while preserving the existing reasoning
loop and tool executor.

The default timeout is 20 seconds per interaction/extraction/image candidate,
with a 40-second total deadline. Reasoning/planning/judging/coding default to
120 seconds per candidate and 180 seconds total. Operators may adjust these
bounded limits per role. Source assertion and image workers also enforce their
own outer bounds. A timeout, connection failure, rate limit,
context-window exception, completion 404 or transient server error can try the next eligible
candidate. Authentication, validation and redirects do not trigger blind
retry. The same endpoint/model/weight generation is not retried under another
legacy alias. `allow_fallback: false` restricts a call to its first candidate.
Explicit legacy `force_tier` remains exact and cannot bypass eligibility.

An observed endpoint/model failure cools down for 15 seconds, so subsequent
requests can use an eligible fallback immediately. After that interval, one
request attempts recovery while concurrent requests keep using their eligible
fallbacks. A successful completion restores the primary. Cancellation releases
the recovery slot. If no eligible fallback is available, the request reports
that fact. Updating the endpoint or model configuration starts new observations
without a restart; an in-flight call still retains its original snapshot.

The caller's request deadline does not start that shared cooldown. A short
extraction allowance says nothing about whether the same model can complete
a longer reasoning request. Caller expiry is recorded as `RequestBudgetExceeded`;
an actual transport `TimeoutError` remains an endpoint failure. The fallback
list is still visited once within the role's total deadline. An endpoint that
only stalls until the caller timer expires remains eligible for later requests;
no extra retry, adaptive deadline or throughput estimate is introduced.

A fallback must retain the request's modality and configured capability
constraints. The existing text token estimator also excludes obviously
undersized declared contexts, including the requested output allowance. This
is a heuristic, not exact tokenizer accounting; image token usage and unknown
legacy context sizes remain unknown.

All automatic function candidates must be local. Default networks are loopback,
RFC1918 IPv4 and IPv6 unique-local ranges. `localNetworks` can replace these with
deployment CIDRs. A hostname must also appear in `localHosts`, and every resolved
address must fall within the configured networks. The resolved address is
pinned for that request. A friendly hostname suffix is insufficient. Requests
use explicit endpoint credentials, bypass environment proxy configuration and
do not follow redirects. Model endpoints themselves remain deployment-trusted
services. A prompt or `allow_cloud` hint cannot authorize cloud fallback; a
separate explicit authorization policy would be needed for that behavior.

A hostname with an unavailable address family can try its other verified
addresses after a connection failure. At most eight unique resolved addresses
are considered under the same candidate deadline; authentication and validation
responses do not cause address retries. This also applies to model discovery.

## Observable behavior and provenance

`GET /v1/host/models` observes the explicitly configured pool through each
endpoint's OpenAI-compatible `/v1/models` route. Both a server root and an
existing `/v1` base are normalized correctly. Reads reuse observations for 30
seconds and probe at most four endpoints concurrently, with a two-second bound
per read. Additional reads cover larger pools, starting with unobserved or
oldest endpoints. No timer, scanner or background service is added.

Routing status separates model advertisements from actual completion outcomes.
It includes advertised model IDs and context limits, observation timestamps and
ages, stale flags, inventory completeness, completion latency, served model ID
and cooldown/recovery state. An advertisement never grants tool or vision
support or changes declared context and performance constraints. A configured
request alias can work even when absent from the server's listing. Only an
actual failed completion affects availability. A missing listing therefore
remains an observation failure without disabling working inference.

Source assertions default to `extraction`, with an optional task-role override;
image descriptions request `vision`,
and project plans request `planning`. Existing named task hints map
ToM extraction to `extraction`, working-state thinking to `reasoning`, skill
distillation to `judging`, and tool drafts to `coding`. Other router calls default
to `reasoning`; this does not alter Hermes's separately selected chat model.

Each successful response names its function, selected binding/model,
configuration revision and `weightRevision`. Missing immutable model revisions
remain `unknown`. The configuration fingerprint identifies supplied config,
not model weights or an attestation. Assertions retain these fields in their
existing derived JSON. Image descriptions retain them in their existing media
row. Reasoning results pass them into existing operational outcome evidence.
`GET /v1/host/models` includes routing metadata and up to 20 recent successful
call traces without prompts, endpoint URLs or credentials. These recent traces
are process-local diagnostics; persisted source/operational evidence remains
the durable provenance.

Qualification uses real disposable HTTP endpoints for failures, candidate
constraints, declared-host resolution, in-flight reload, retained router
references, source assertion projection and image caption storage. Two neutral
actual-LAN calls also used the same retained router before and after a temporary
configuration change; the second recovered from a closed local endpoint and
selected the new binding. This is functional qualification, not an intelligence,
latency population or extraction-quality benchmark. Production adoption still
requires a real deployment config change and observed downstream behavior.
The endpoint observation extension also exercises URL normalization, working
unlisted aliases, completion 404/503 fallback, concurrent recovery, cancellation,
cache expiry, pool coverage and an endpoint move through atomic config reload.

## Usage: model calls by lane

Every model call the sidecar makes leaves one row of counts in `usage.db`, in
the instance directory: when the call ended, the client that made it, the
caller's task, the role, the instance's own name for the endpoint, input and
output tokens, latency and the outcome (`ok`, `error`, `timeout`,
`incomplete` for an answer that came but could not be used, `cancelled` when
the caller stopped waiting, as a decision tier does at its deadline). A
row never holds the words of a prompt or an answer, an endpoint URL, a model
name, or an id of a person or a source. A task is recorded only as a code
name (lowercase letters and underscores) or as the name of a decision point
the process registered (the mind's points, and each external point of
`instinct.external`, whose names may hold digits); any other task is recorded
as `unnamed`. A role or an endpoint is recorded as it is when it is a plain
name (a letter, then letters, digits, `_` or `-`). Any other name, such as a
binding named `worker.v1`, a host name or an address, is recorded as `alias-`
and the first 12 hex digits of an HMAC-SHA256 under the instance's own secret
(`instinct.key` in the instance directory, which a backup carries only when
it is encrypted), taken over the name's SHA-256 prefix: two names are never
merged, none is kept as text, and the ledger alone cannot be searched for the
address behind an alias (the same name has another alias in every instance).
`colonymind report usage --alias NAME` says what this instance records for a
name. A row whose alias cannot be keyed is dropped, never kept unkeyed. An
earlier build kept that SHA-256 prefix itself as the alias; the first write
after the upgrade keys those aliases in place, once, so a name's earlier rows
and its new ones stay one group. Until that write, `--alias` names both: the
alias the earlier rows still hold and the one the write gives them. A secret
file that holds no whole key keys nothing (those rows are dropped), and
`colonymind doctor` fails it (`instance-secret`) with how to recover. Tokens are
the counts the endpoint itself reports, input and output each on its own: a
count the endpoint did not report is none, not zero, even when it reported
the other. An embedding or a rerank generates no tokens, so its output count
is 0.

| Client | One row per | Task | Role | Endpoint |
|---|---|---|---|---|
| router | attempt, so a fallback's failed attempt is its own row | the caller's task | the function role (or the tier on the tier path) | the binding's name |
| instinct | question a decision server is asked, or completions call of the reader (one per rotation of the options) | the decision point | the provider's name in `instinct.providers` | the same name |
| embedder | attempt at a remote endpoint, or call of a local model | what the embedding is for (below) | `embedding` | `embed` |
| reranker | call, remote or local | `rerank` | `rerank` | `rerank` |

A local embedder or reranker (on the CPU, CUDA or MLX) is covered like a
remote one: a row per call with its latency and outcome. A local model
reports no token count, so its rows have none, never a zero.

An embedder cannot tell from its own call what an embedding is for, so the
caller names it: the text of a captured source is `source_embedding` (a
memory lane), re-embedding a source already indexed, for a new embedder's
index, `source_reindex`, and a warmup or a health check `embedding_check`.
Where no caller names one, a live query is `query_embedding` and any other
text (the embed and index routes, a rebuild of the stored vectors, the mind's
claim keys) `text_embedding`. Only `source_embedding` counts toward the
memory lanes' cost per captured message, however often the health route
checks the embedder.

Not covered: image captions and image embeddings (the multimodal lane), the
admission proxy of the inference pool (it relays another program's calls),
`colonymind demo` (its own scripted model) and the benchmark and qualification
harnesses when they call a provider directly.

A call hands its row to a queue and goes on; one writer thread appends the
rows, so recording costs a call no disk time, and a ledger that cannot be
written, or a full queue, drops the row and never fails the call. Rows are
written only to the state directory the process names
(`COLONYMIND_STATE_DIR`, else `COLONYMIND_HOME`, as the sidecar's configuration
sets them): a script or harness that names none records nothing. Rows and
capture counts older than 30 days are dropped by the sidecar's daily
maintenance, whether or not the instance still makes model calls.

The health route (`/v1/host/health`) carries the last 24 hours' totals as its
`model_usage` note: calls, how many ended ok, and tokens in and out, each a
lower bound ("at least") when some calls did not report it.

A ledger that cannot be read (held locked past the readers' five-second wait,
or damaged) is never read as no usage: the health note says reading it
failed, with SQLite's reason, and `colonymind report usage` prints the reason
and exits 1. Only a ledger not created yet, or with no table yet, reads as
none recorded.

`colonymind report usage [--since 7d] [--json]` reads the table: calls,
failures by outcome, tokens and p50/p90 latency per task and role and per
endpoint, the last 24 hours' totals, and the memory lanes' cost per captured
message (claims and their review, commitments, appraisals, media descriptions
and the embedding of a captured source's text, over the user messages captured
in the same window). Captures are counted in `usage.db` too, as each person-scoped
source is captured: a time and a number of user messages, nothing that names
the source or the person. Erasing a source later never changes that count,
since its calls were made whatever became of its words. Input and output
tokens are estimated apart: a memory call whose endpoint did not report one of
them makes that side's per-message estimate `partial` (a lower bound), or
`unknown` when no memory call reported it; it is never counted as zero. `--json` gives the same
numbers with sorted keys and a fixed shape, for the cost estimates of
re-extraction and the endpoint budget.
