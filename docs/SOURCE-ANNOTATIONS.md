# Attributed corrections to retained source evidence

`POST /v1/host/memory/sources/annotations` appends a correction to an exact source
revision without rewriting the original. The route is on the owner's side of the API:
a credential reaches it only with `manage` in its access, which by default only a
credential granted the owner has, and a credential carried from 1.9.0 needs
`memory:write` ([credentials](CREDENTIALS.md#the-owners-side-of-the-api)). The source
must be one the named person's recall could show: a person-scope source of that
contact, or a session source of the session named. The server records the
authenticated principal and the recording time. Request text cannot claim another
author or turn an ordinary message into an authorized correction.

```json
{
  "contact_id": "example-person",
  "session_id": "later-review",
  "annotation_id": "review-1",
  "source_id": "example-report",
  "source_version": "<64 lowercase hexadecimal characters>",
  "excerpt": "Verification occurred at 09:14.",
  "correction": "The digest comparison is supported, but its verification time was not measured. The reported time is unsupported, not disproven."
}
```

The source reference comes from a card of the memory inventory (below: its `source`,
with the card's `quote` as the excerpt), from the source references recall returns,
or from a memory search's results. No Hermes tool writes a correction here: in
conversation a correction is the person's own statement, and it forms claims like any
other ([claims](SOURCE-CLAIMS.md#corrections-changes-and-conflicts)).

`annotation_id` is the caller's idempotency key, scoped to the person and the
authenticated principal: an exact replay returns `created: false`, and different
evidence under the same key conflicts (`409 annotation_id_conflict`). The excerpt must
occur exactly in a retained text message of that revision; excerpt and correction are
each limited to 4,096 characters, and a secret in the correction is sealed as in any
source. A stale revision or an excerpt that is not there conflicts (`409
source_version_mismatch`, `source_excerpt_mismatch`); a source the person's scope does
not hold is refused (`422 source_not_found`), and so is a key whose earlier correction
was erased (`422 source_erased`): it is never written again. The response returns the
annotation's `source_id`, `source_version` and exact `target` reference. A confirmed
correction remains attributed evidence, not verified truth.

The annotation is a new assistant evidence source in the existing source ledger,
with a relation to its target committed in the same SQLite transaction. It does
not create a USER assertion or assert that the correction is independently
verified. Existing source dependencies link the annotation to the target.

Lexical and semantic candidates, source-backed belief/assertion bundles, and
answers with recorded supplied-source ancestry receive the applicable corrections
before ranking. The original evidence and full attributed corrections form one
atomic packet. If it cannot fit the configured context budget, that packet is
omitted. Different original excerpts remain distinct; an annotation-only search
hit does not repeat an already supplied correction bundle. Multiple corrections
remain visible without treating the newest one as automatically true.

Selected packets are checked again after ranking for changed source revisions or
annotations relevant to their actual source messages. Direct quotations retain
their message identity; atomic claim bundles carry the canonical claim message
hashes. An independent assistant sibling contributes neither its dependencies nor
its corrections to the recalled message. Erasure of another message does not hide
an unaffected claim. Existing supplied parent references and older graph summaries
remain conservatively bound to whole source versions when no finer provenance was
recorded. An annotation's own target relation retains its exact message membership.

A turn's memory section carries the original and the correction with exact source
references for both, so an answer captured from that turn keeps both dependencies.

Forgetting either source invalidates dependent answers through existing erasure
rules. Erased annotation text is removed; the content-free relation remains so an
unqualified original does not silently return. Partial source erasure never
rebinds the correction to another revision, and unrelated surviving message
evidence remains available. History without supplied-source lineage cannot be
inferred to depend on an annotated source.

Forgetting also reaches the person digest (`docs/CONSOLIDATION.md`, per-contact digests). A digest
records the claims it was built from; one that cites a claim an erasure removed, or a claim a
correction retracted, is withheld from the person's context, the owner's inspect and every host contact
response, its text dropped from the contact's record, until it is rebuilt from what is left without
reading the old text back. The forget response reports `digest_cleanup` beside the other cleanups.

This is an attributed source correction, not an automatic fact verifier, topic
matcher or model judge. Text outside canonical source evidence and non-text
messages cannot be annotated through this route. Older backends retain the
additive rows but do not implement mandatory correction expansion; that guarantee
requires the annotation-aware backend.

## The memory inventory

`GET /v1/host/memory/inventory?person_id=<id>` (or `colonymind memory inventory <who>`) lists what memory holds
about one person: their source claims, newest first in the order they were formed, one card each. A card has the
claim's `claim_id`, `state`, `subject`, `predicate`, `value` and `quote` (the exact words it was formed from), when it
was said and recorded, its validity, what it changed or corrected (`prior_claim_id`), what superseded or retracted
it, how many attributed corrections it carries, and `source`: the `source_id`, `source_version`, `session_id`
and `message_hash` of the message it quotes. That source reference is what the annotate route above takes (with the
card's `quote` as the excerpt) and what forget takes (`source_ids`).

States follow the claim's dates, as recall reads them:

- `current`: live, and nothing recall reads disagrees with it.
- `in_conflict`: recall shows it beside a live statement of a different value for the same property (literal values
  that differ, with overlapping validity; quoted preferences are never compared), or a change to it waits for the
  owner.
- `upcoming`: its stated validity has not begun (a change dated ahead, until that date).
- `expired`: its stated validity has ended.
- `superseded`: a change replaced it, and the change's date has come. Until then the claim it replaces is live
  (`current` or `in_conflict`), and its card already names the change (`superseded_by`).
- `retracted`: a correction, or the owner's answer to a question, withdrew it.

An attributed correction does not change a card's state: the card counts the corrections that name a message stating
the claim (`corrections`): its quoted message, and an earlier statement a repeat folded into it (an unfolded store
kept that statement as a claim of its own, with the correction), and the corrections recorded against those
corrections. Recall presents each with the words it was recorded against, and the history shows their words. Every
claim is formed and reviewed by a model, so a card is `verification: unverified` unless the owner kept that claim in
an answer (`owner_confirmed`). The owner's answers about a claim stay as long as the claim does: the learning log's
nightly bound on its own entries never drops them. Claims are read as recall reads them: the person's own scope,
current attribution, the quoted message still retained. An erased claim is absent, and so is every claim of a source
an erasure touched.

Who may list whom: a credential that manages the instance lists anyone it may name; any other credential lists only
the person it is bound to (its viewer, its listed contacts, or its own identity). Both routes are memory reads, like
search: a credential bound to a person reaches them with its default access. Development mode (no key) lists no one.

Pages hold still under writes. `limit` is 50 by default and at most 200 (a builder's choice). A page's `next_page` is
passed back as `page`. A listing covers the claims that existed when its first page was read: a claim formed since
appears in the next listing, never between pages, and a correction or change shows on the page that is read after
it. Its `read_revision` covers the person's erasures, attribution changes and removed claims; a next page asked for
after any of them is refused (`409`, `inventory_changed_restart`), and the caller drops the earlier pages and lists
again. `erasure_watermark` is the person's erasure sequence the listing began at.

`GET /v1/host/memory/inventory/claims/{claim_id}/history` (or `colonymind memory history <claim>`) walks one claim's
own chain, first statement first, the same from any claim of the chain. Its steps:

- `stated`: the first statement, with its card;
- `superseded` and `retracted`: which claim ended which, with the card (and so the source reference) of the
  statement that ended it, or the owner's answer (`by.ref`, `by.owner_answer`);
- `corrected`: an attributed correction of a claim, with its words, author and source, and the source it was
  recorded against (`corrects`): the claim's own message or an earlier statement a repeat folded into it, or another
  correction (one that withdraws or qualifies it, followed as recall's expansion follows it);
- `kept_by_owner`: the owner kept the claim in an answer;
- `preference_change`: a change to a quoted preference held for the owner, and what became of it (`status`:
  pending, ratified, declined). It joins the two statements whatever the owner decided, so the history is the same
  from either; it ends the earlier one only once ratified.

An erased claim or correction is a content-free marker (its id and `erased: true`), never its words; the chain
continues past an erased claim through the links other claims keep to it, and a correction recorded against an erased
correction is not read (it quotes the erased words). A claim that names an erased claim as what it changed or
corrected links to it only when that took effect: a correction, or a change with a date, and not a change held for the
owner that the owner did not ratify (its pointer only names what it proposed). `complete` is false when the chain may
go on past what is shown: the walk reads at most 64 claims (a builder's choice; `limit_reached`), and an erased
claim's link on one side went with its row unless a retained claim keeps it (`gaps`: each such claim, with the side,
`before` or `after`). An erased claim is never taken for the chain's first or last statement, and two parts of a chain
that no retained claim joins are never joined. `person_id` is optional: without it the claim is read in its own
person's scope, under the same rule, and a claim the credential may not read is not found (`404`), never refused in a
way that tells it exists.

Both are reads of existing records, never a new store, and they write nothing: changes still go through annotate
and forget. The routes answer with the stored words; `colonymind memory inventory` and `history` (text and `--json`)
and the owner console print them read as a card shows them, the predicate beside the value: a value the capture
detector would seal there, or in any of a claim's words ("wifi password: ..."), is withheld wherever they print it,
its quotation and corrections included ([the sealed store](SECRETS.md#claims)). No native tool reads the inventory, and listing a card never counts as supplying its source to a turn.
Listing never widens annotation either: the annotate route takes the owner's side of the API (or, for a credential
carried from 1.9.0, `memory:write`), so a credential that may list a person's cards is still refused there.
