# Voice affect tail read (staged)

`the voice client's affect URL setting` targets `POST /v1/host/memory/voice-affect` on the
speaker's own ColonyMind instance. Send the same bearer credential or `X-API-Key`
as voice recall and this body:

```json
{"identity": {"host_id": "companion"}, "person_id": "owner-contact-id"}
```

The credentialed Companion relay supplies the actual resolved current speaker,
as it does for voice recall. The speaker ID is required; there is no implicit
owner default. The response has exactly one field:

```json
{"line": "Sam: frustrated, low energy · Nova: interested"}
```

Append the nonempty line to the dynamic per-turn tail, as specified by
`VOICE-LANE-DESIGN.md`'s **Emotion tags** section. The client applies a **20 ms
deadline from request start**: timeout, failed request, or late response means
no line. The client must never wait for a retry or infer a replacement. The
server also discards the line if its handler has already taken 20 ms, though
this check cannot preempt a blocked synchronous SQLite read. Qualification is
local warm serial latency, not a guarantee under locks, cold imports or live
transport. Client configuration and activation belong to the serving integration.

Authenticated requests with no available permitted state return HTTP 200 and
exactly `{"line": ""}`. This includes absent, stale, malformed or unavailable
state, identity configuration conflicts, privacy exclusions, and elapsed
handler deadlines. Invalid/missing credentials and credentials without read
access retain normal authentication refusals; invalid request shapes retain
422. The client treats either case as no line.

## Fixed mapping and caps

Use the existing `AffectStore.get_state(person)` projection, including its
canonical source erasure checks, weighted history, and decay toward neutral.
The read can persist decay and clean erased projections as existing state
readers do. This unit adds no classifier, affect store, model inference,
embedding, vector lookup, appraisal processor, or speculative job.

| Input | Bin | Plain output |
|---|---|---|
| Current valence | -1 to -0.25 inclusive | frustrated |
| Current valence | greater than -0.25 and less than 0.25 | neutral |
| Current valence | 0.25 to 1 inclusive | pleased |
| Current arousal | 0 to less than 0.35 | low energy |
| Current arousal | 0.35 to 0.65 inclusive | steady energy |
| Current arousal | greater than 0.65 to 1 | high energy |
| Own appraisal dimension | frustration | frustrated |
| Own appraisal dimension | annoyance | annoyed |
| Own appraisal dimension | interest | interested |
| Own appraisal dimension | satisfaction | pleased |

Every fresh speaker descriptor is `<valence word>, <energy words>`. A fresh
observed neutral state can say neutral; an absent default neutral state cannot.
These are deterministic bins, not diagnoses or inferred causes. Appraisals
use the existing `self_model/appraisals.py` store, not contact valence. Only
the newest current appraisal head for the owner's subject is considered.
If that head is expired, stale, invalid or unsupported, omit Nova's part
without falling back to an older mood. Preferences, hypotheses, assessments,
settled/withdrawn/reconsidering records, internal text, topics, support,
intensity, hints, and numerical scores never become the line. The store does
not have an amusement dimension, so the example word **amused** is not emitted.

The speaker and Nova parts are independently optional, joined by ` · ` only
when both are present. The line is **at most 120 characters and 25 tokens**
by `memory.recall.estimate_voice_tokens`, the existing cautious voice counter.
No square brackets, em/en dashes, or line breaks are permitted. Configured
owner/persona names or the verified contact's stored name can label a part;
labels allow only letters, spaces and apostrophes, with a 20-character limit.
Unsafe labels become `Speaker`/`Assistant`. If the combined line exceeds a
cap, first use those short labels, then keep only a complete speaker descriptor,
then only a complete appraisal descriptor, or return empty. Words are never
truncated. Response schema validation also enforces both caps and forbidden
characters. Operator token-counter overrides cannot relax the cautious cap.

## Freshness and privacy

**Stale means age at least 1,800 seconds (30 minutes)**. Speaker freshness uses
the latest non-decay event's observation timestamp, never `last_updated`:
decay and repeated reads cannot freshen old evidence. Absent observations,
future timestamps, naive/malformed timestamps, and non-finite or out-of-range
valence/arousal omit the speaker part.

An own appraisal must have current status, remain an active head, be strictly
before its store expiry, and have both creation and observation ages below
30 minutes. A delayed background interpretation cannot freshen an old source.
Legacy records without an observation clock use the canonical source's aware
occurred/ingested timestamp. Dependencies must still match canonical source
versions, speaker attribution and message hashes, including source erasure
visibility. Only a fixed dimension leaves the internal reader.

This is the same **read** operation as voice recall, including the legacy
`memory:search` scope. It adds a stricter speaker privacy filter:

* Owner speaker: explicit person ID must match the configured instance owner,
  and the credential must grant owner platform `voice` (or every platform).
* Verified contact: the credential must be configured for `contacts: own` or
  a contact list with a bound `viewer`; the speaker must equal that viewer.
  An unbound credential or a broad `contacts: all` relay cannot inspect other
  contact affects, even if ordinary recall permits selecting them.
* A guest cannot receive the owner's affect or Nova's private appraisals.
  Guest requests never open the owner appraisal reader. An owner relay carrying
  a guest's person ID receives empty. Managing access confers no exception.
* Development mode receives empty, and conflicting/missing owner identity
  fails closed. The credential binding is the verified identity; names, body
  fields, or supplied sender labels are never used to prove identity.

Synthetic examples: `Sam: frustrated, low energy · Nova: interested`,
`Sam: neutral, steady energy`, `Nova: pleased`,
`Sam: pleased, high energy`, and the empty string. Deployment names come from
the existing identity configuration; the unit does not hardcode personal names.

## Offline qualification

The new gate seeds **10,000 affect events, all in the measured owner's scope**
(64 source-linked events plus imported unlinked history), **64 real canonical
sources**, **10,000 appraisal records**, and **64 active heads**. Controlled
history SQL fixtures use production schemas; canonical sources use the normal
ingestion writer, and the affect projection is recomputed by `AffectStore`.
Warm 20 authenticated calls, then time 300 serial `httpx.ASGITransport` POSTs
using `perf_counter_ns`. The interval includes credential middleware, request
validation, actual store readers and JSON response completion. p50/p95 use
nearest rank. Every response must contain both parts; an empty/late response
cannot satisfy the latency gate. No socket is opened. Counts, sample lines,
timings and the gate decision are written to `VOICE_AFFECT_REPORT`.

This qualifies that mixed seeded history, not 10,000 independently source-linked
events or concurrent traffic. The bounded payload is independent of history
size, while existing canonical projection validation scales with sourced
history. Detailed staged measurements and test counts are recorded in
`VOICE-AFFECT-out.md` and `voice-affect-measurements.json` after qualification.

`$PYTHON` is the Python of a ColonyMind development environment and `$WORKDIR` any scratch directory.

```bash
TMPDIR=$WORKDIR \
COLONYMIND_TEST_POLL_LOOP=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VOICE_AFFECT_REPORT=$WORKDIR/voice-affect-measurements.json \
$PYTHON -m pytest -c sidecar/pytest.ini \
  sidecar/tests/test_voice_affect.py sidecar/tests/test_voice_affect_gate.py -q -s
```

Existing tests and fixtures are unchanged. New tests cover all bins and their
boundaries, response constraints, source erasure/correction, independent missing
parts, freshness, decay, privacy and credential parity, no inference, malformed
state, and fail-empty behavior. The two-failed-round stop rule applies.

Staged result: **p50 7.23 ms, p95 8.56 ms, max 10.45 ms**; all 300 timed
responses contained both parts. The revised round passed **645 tests**:
144 new contract/unit tests, one new seeded gate, and 500 unchanged regressions.
Two skips: credential-grant collection imports stock Hermes, which is absent
from the supplied venv; the optional real-embedding voice gate requires an
external vector JSON not supplied to this run. The unchanged affect (26),
source-appraisal (53), voice recall (270), voice review (87), speculative voice
(33), original voice gate (1), public API (27), and authentication coverage (3)
tests all passed. New tests exercise read grants and legacy scope parity too.

Round 1 passed 118 and failed 21: the frozen fixture clock had more precision
than persisted ISO timestamps and could make fresh observations slightly
future-dated. Round 2 uses an exact-second fixture clock; freshness assertions,
performance thresholds and existing tests were not weakened. Only one round
failed, so the two-failed-round stop rule was not reached.

Implementation and validation files:

* `sidecar/colonymind/api/credentials.py`
* `sidecar/colonymind/api/legacy_scopes.py`
* `sidecar/colonymind/api/routers/host.py`
* `sidecar/colonymind/api/schemas/host.py`
* `sidecar/colonymind/contacts/affect.py`
* `sidecar/colonymind/self_model/appraisals.py`
* `sidecar/colonymind/memory/voice_affect.py`
* `sidecar/tests/test_voice_affect.py`
* `sidecar/tests/test_voice_affect_gate.py`
* `docs/VOICE-AFFECT.md`
* `docs/VOICE-AFFECT-out.md`
* `docs/voice-affect-measurements.json`
