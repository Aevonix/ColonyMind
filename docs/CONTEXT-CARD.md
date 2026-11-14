# Daily context card for voice (staged)

The card targets 400 tokens of supported context, with an 800-token conservative cap.
Sparse cards may be shorter than 400; the builder never repeats coverage guidance or invents
content to fill the target. It contains the configured owner name, recorded communication
preferences and priority corrections, the assistant's self-narrative, observed work,
recent direct-channel excerpts, and an already delivered owner daily digest when available.
The assistant's narrative is labelled separately from the owner's identity.

The deterministic builder makes no LLM calls. It bounds input fields and selected items,
keeps a fixed section order, labels shortened excerpts, and uses a colon in the dated header.
Em and en dashes in producer text are normalized to ASCII hyphens. The cap estimator
`context_card.conservative_tokens.v2` counts Latin/ASCII runs with the existing router
estimator, rounded up per run. Each other codepoint costs at least one token. Emoji bases
cost at least three tokens each, including every base in a ZWJ or flag sequence; joiners,
variation selectors and modifiers receive additional allowances. This deliberately
conservative estimate is not a measurement with the serving engine's tokenizer.
Synthetic stress fixtures matching the review's old estimates (757 for CJK and 795 for
emoji) exceed its supplied GLM measurements (1398 and 1958) under the new count. No GLM
model or tokenizer is downloaded by the offline gate.

The collector reuses `PreferenceLearner.build_brief`, `SelfPerspective.brief`, `Mind.narrative`,
`Mind.section`, `CommitmentStore.list`, `read_recent`, and the
initiative store's delivered digest. Mind, preferences, commitments, initiatives and bounded
communications locator metadata are captured synchronously on their owning event loop.
Only detached values and ledger reads go to the worker thread. Legacy communications
locators are queried through a thread-local in-memory snapshot, never a shared connection.
Open source-backed obligations are withheld when their lineage is unavailable or erased.
Work is observation metadata, not a claim that execution is complete. Highlights retain role
and channel labels. The canonical reader enforces person/session scope, occurrence time,
corrections, and erasure. `api_server` automation and raw pendant/observed-conversation sources
are omitted, including device sources incorrectly labelled as direct channels. A stored
digest predating an owner erasure is omitted.

The lifecycle worker is **disabled by default**. Its staged configuration is:

```yaml
context_card:
  enabled: false
  poll_seconds: 5
  reconcile_seconds: 300
```

The worker checks metadata for input stores, WALs, identity/config files, journal cursor,
owner binding, estimator setting, and local day. It coalesces commits between polls;
periodic reconciliation also covers in-memory updates and time-dependent producers.
Unchanged revisions skip collection, and unchanged normalized inputs skip rendering.
Only changed owner/card bytes advance the version. The owner resolver is cached per
configured binding, so repeated fetches do not repeat legacy deprecation warnings;
environment binding changes invalidate it immediately.

GET serves the cached card. Ordinary watched-metadata changes and refresh failures keep
the previous card available with `stale: true`, including local-day changes. Publication
is postponed for at most two consecutive passes with racing ordinary writes, then
publishes the collected evidence as stale and schedules another reconciliation. The
postponement limit never bypasses owner or erasure checks. Startup withholds persisted
bytes until reconciliation. Owner rebinding or a changed erasure watermark immediately
withholds the card, including conditional requests; a racing erasure/rebind prevents
publication. The privacy check reads only the owner erasure head through a separate
read-only SQLite connection when ledger/WAL metadata changes. It never initializes a
source store or hydrates source bodies in GET. An unreadable privacy head fails closed.
The collector also checks the erasure head before and after its ledger pass.

`context-card.db` uses private SQLite/MindState primitives, an atomic current pointer,
and only the current version row. Opening an older database prunes its historical rows.
Owner rebinding and erasure purge derived rows; an unrelated erasure that leaves bytes
unchanged preserves the version. Version sequencing survives purges and restarts.
The disabled lifecycle deletes `context-card.db` and its WAL, SHM and journal files.

Before enabling, an operator can time one collection pass on a quiescent snapshot:

`$PYTHON` is the Python of a ColonyMind development environment and `$WORKDIR` any scratch directory.

```bash
cd sidecar
TMPDIR=$WORKDIR $PYTHON \
  -m colonymind.context_card.measure /path/to/state-dir-copy \
  --home-copy /path/to/instance-home-copy --owner-id cid-owner --day 2026-10-14
```

`--home-copy` and `--day` are optional. The script clones the supplied snapshot into
TMPDIR, opens producers only in that disposable clone, captures shared state on its
own event loop, and performs ledger reads in a worker thread. It prints setup, shared
capture, ledger-read, total collection and build milliseconds, conservative tokens,
and coverage counts without card text. Constructor/migration time is reported separately
from the single collection pass. It blocks network connections and model downloads,
starts no service or Mind tick, publishes no card, and leaves the supplied snapshot intact.
Snapshot integrity and matching owner/configuration are the operator's responsibility.

Companion fetches and revalidates off the utterance path using an owner reader credential.
It keeps current/pending versions, composes the stable voice prefix, and prewarms through
its serving integration. A warm acknowledgement permits choosing that version for new
voice generations; ongoing generations retain their version. `ready_for_prewarm` means
the bytes are published, not that any model cache is warm. Companion assembly,
prewarming, activation and time-to-first-token measurement belong to the serving integration.

The offline gate includes `sidecar/tests/test_context_card_review.py`,
`sidecar/tests/test_context_card.py`, and `sidecar/tests/test_public_api.py`.
