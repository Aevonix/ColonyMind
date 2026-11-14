# Working memory: branches and consults

A finished worker's message text and tool results can remain available as a branch.
The parent receives the worker's normal result and a short branch handle. When
the result leaves out a detail, `colonymind_consult` reads the named branches
together and answers the question with span handles. Only owner sessions can
capture, list or consult branches.

The working-memory [handback experiment](../benchmarks/working_memory/README.md)
earned this text path. The host hooks carry message lists, not the child's system
prompt or tool schemas. A consult therefore reads ColonyMind's canonical render
of those messages. It does not reproduce the child's exact prompt or attach any
KV state. Repeated reads of unchanged material send the same prefix, which a
serving engine's ordinary prefix cache may reuse.

## Capture and context

The Hermes plugin queues the completed child's messages using the parent
session's trusted sender identity. Guests' and contacts' children are excluded.
The API independently checks the credential's owner-platform grant and resolves
the sender to the owner. The local CLI can use its granted owner platform with
an empty sender id; another platform needs a known owner handle. A requester's
claimed name alone grants no access.

Every retained delegation brief, message and tool result becomes a ledger source
with a `branch_span` content block, owner audience and `derived_unverified`
status. Capture safety runs before a write. These blocks never supply text to
ordinary claim derivation, lexical recall, vector indexing or projections.
Worker material remains unverified material, including anything a tool returned.

Hands keep that branch boundary. The [hands provenance APIs](HANDS.md#provenance)
create an owner-audience entry only for an accepted result at E3 or above,
embedding its check receipts. Captured assistant relays carry the shown hand's
evidence level and recall renders its marker; repeating an unchecked result
does not verify it or teach a lesson. Branch expiry keeps embedded support but
downgrades relay markers. Parent erasure retracts dependent entries and lessons,
and an owner reversal disputes the result. The controller and retention callers
must invoke these lifecycle APIs as described in the integration notes.

A manifest holds its handle (`br-` followed by twelve hexadecimal digits), parent
and child sessions, creation time, audience, and ordered source ids and content
hashes. The parent's next context receives a **Branches** section with each
handle and the brief's first line, bounded to 120 characters (implementation
assumption). Before an owner conversation is compressed, a session branch can
reference its existing captured turn records without capturing another copy.

Branches and their own spans expire after 14 days by default (implementation
assumption).
An hourly retention sweep (implementation assumption) runs even with the faculty
off; branch reads and writes also remove expired material before serving it.
Erasing or changing a span removes it from future renders; any remaining source
text keeps its original expiry. Erasing or changing a parent session record
conservatively removes that session's worker branches, including when the parent
record arrived after the child. Handle-only tombstones prevent a capture retry
from restoring those branches. A branch with no remaining spans is removed.
Existing turn records referenced by a session branch retain
their ordinary lifecycle; branch expiry does not erase the conversation's
original records. Host-owned transcripts still have their own erasure boundary,
described in [Known gaps](KNOWN-GAPS.md).

## Consults and limits

`colonymind_consult` accepts `handles` and `question`. The adapter supplies the
session and sender outside the model's arguments. The sidecar verifies each
manifest's audience, sorts branches by handle, renders all of them in one
request, then appends the question and an instruction to answer only from that
material and cite span handles. The request has no tools. Questions spanning
children use one combined read.

The `consult` router task defaults to the `reasoning` function role. A deployment
can map it through `taskRoles`, for example:

```json
{"taskRoles": {"consult": "reasoning"}}
```

That role's configured models, context capacity and deadlines apply. See
[Function routing](FUNCTION-ROUTING.md).

The default input cap is 48,000 tokens and the hourly instance budget is 20
consults (implementation assumptions). Admission counts UTF-8 bytes as a
conservative upper bound on tokens, including the question and fixed instructions, with an
additional 1,024-token chat framing allowance (implementation assumption). It can
therefore refuse a request below a model's actual 48,000-token capacity. An
over-cap refusal names the requested handles. The sidecar does not silently omit
a branch. A spent budget refuses another consult. Admitted attempts count toward
the budget even if the model call fails; the sampled summary call shares that
attempt's allowance. Endpoint-reported cached tokens remain usage evidence only:
identical prefix text does not guarantee that a serving cache kept it.

## Comparator and withdrawal

A keyed hash selects one consult in ten for a second answer from the worker's
summary alone (implementation assumption). Both answers use the same router
task. The comparison log stores keyed answer hashes and references, not answer
text. The owner's next turn is read by the existing reaction reader. Its
explicit positive or negative reaction labels the served answer; a reaction
that the reader cannot recognize leaves it unlabelled.

A served-answer reaction does not establish whether a different summary answer
was right. A complete labelled pair needs the owner's separate explicit reaction
to that reference answer. On a sampled consult the tool returns both answers and
explains the optional next-turn rating format, a JSON object with exactly these
fields, for example `{"answer":"Not helpful.","summary":"Keep going."}`.
Each value is read by the existing reaction reader. When both answer hashes are
identical, one reaction applies to that same answer in both arms. This measures
explicit owner preference, not independently verified factual correctness.
Without both labels the comparison remains partial. A turn following multiple
pending consults labels none, since its reaction cannot select an answer
unambiguously. A session branch has no worker summary, so its consult cannot
produce a summary comparison.

The decline check uses up to the newest 200 labelled pairs (implementation
assumption) and the existing plasticity `gates.decline_bound` bar. It can withdraw
consults before 200 pairs accumulate when the lower 95% decline bound exceeds
the bar. Withdrawal records the reason. An operator clears it with
`colonymind wm reset` after reviewing the report; reset starts a fresh comparison
window while retaining historical totals.

## Controls and report

`mind.faculties.working_memory` ships enabled. Turning it off stops branch
capture, consult availability and the Branches context section. Existing
branches remain subject to erasure and expiry. The paired benchmark supports
the faculty-off arm and `wm.summary_only`: capture stays on in the summary arm,
while consults are unavailable.
Benchmark images must advertise `paired-working-memory-1` for either explicit
working-memory arm. The worker mounts the branch routes over the same ledger as
its host turns. The faculty report shows summary-only comparisons as mechanism
diagnostics, separately from evidence for switching the whole faculty on or off.

The top-level `working_memory` configuration holds `retention_days` (14),
`token_cap` (48000), `hourly_budget` (20) and `summary_only` (false). The three
limits must be positive integers.

The Working memory section of `colonymind report faculties` reports retained
branch and span counts, lifetime consults and refusals by reason, comparison
evidence, withdrawal state and the consult task's cached-token share when the
endpoint reports it. These totals are not limited to the report's selected time
window. See the [adapter routes](API.md#working-memory-branches-adapter-routes) for the capture,
list, pre-compression, consult and reaction envelopes.

## Deployment check

Install the matching sidecar and Hermes adapter release together, and map the
`consult` task to a deployed role with enough context capacity. After the first
owner delegation, verify that the parent's Branches section names its retained
branch and that a consult cites a span from it. Check a guest's refusal, the
instance budget and the working-memory report. A cache benefit is measured only
when the endpoint reports cached tokens.

The live comparator runs from the first consult. Its report counts complete
labelled pairs; automatic withdrawal uses only those pairs. To roll the feature
back, set
`mind.faculties.working_memory: false`. Capture and tool exposure stop, while
the retained material remains erasable and expires under the retention sweep.
