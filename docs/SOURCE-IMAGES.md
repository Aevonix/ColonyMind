# Durable source images and recalled descriptions

Native Hermes images become local originals, fallible descriptions, and scoped recall candidates across sessions. The existing source reader can reopen an explicitly selected original into supported vision tool-result parts. Description recall and original-pixel inspection are distinct operations. Video timelines, image embeddings and automatic selection of a vision processor remain separate work.

## The source contract

Hermes v0.21 sends native images as OpenAI content lists. The general adapter now retains those lists in the ordinary durable outbox and sends them through `TurnSyncRequest`; it no longer converts them to a Python string. Existing text cognition receives only explicit text blocks. Checkpoints already retain content lists and use the same image storage path.

Automatic retention accepts inline `image_url` or `input_image` data URLs containing a static PNG, JPEG or WebP, at most 4 MiB per image and 32 million pixels. The enclosing source/outbox envelope remains limited to 8 MiB. No URL is fetched and no local path from message text is opened. The transport must supply the bytes. Unsupported attachments become explicit `image_unretained` records. Remote image references retain a digest and reason instead of signed URL credentials. This normalization does not redact a URL independently written in ordinary source text.

The source message stores a content-addressed asset handle. Its original message digest is retained as ledger-owned `_source_message_hash`, so source erasure and host outbox/checkpoint copies still agree. Public message schemas cannot supply that top-level field. The immutable turn digest continues to cover the original input envelope. Current message content and source membership still govern access; a hash is not an authorization grant.

An already captured source is not silently reinterpreted or backfilled. Rolling-upgrade source replay retains its previous digest contract. Existing historical references and unsupported attachment kinds keep their documented limitations.

## Storage and lineage

The existing `LocalImageStore` provides exact-byte storage in the source-owned namespace:

```
$COLONYMIND_STATE_DIR/images/sources/originals/<sha256>.<extension>
$COLONYMIND_STATE_DIR/images/sources/thumbs/<sha256>.jpg
```

Originals are not resized or EXIF-stripped. Thumbnails are separate projections. Files are written with private permissions, an atomic replacement and durability sync before the source transaction references them. A read verifies the original hash. The source namespace keeps cleanup independent from existing vector-store image ownership.

`source_media` and `source_media_links` use the existing canonical turn SQLite ledger. Links identify the source turn, original message hash, block index and role. Descriptions keep the producing model alias and description version. They are `derived_unverified` evidence, not source quotations or verified beliefs.

Descriptions share the existing source projection worker, with one model request at a time. Function routing uses `source_image_description` and the `vision` role, its declared local fallback, and a 40-second outer deadline. Legacy tier routing requires an explicitly configured local VISION tier with image support, a 20-second deadline and no escalation. No SMALL/default binding is borrowed. Unavailable or incompatible local models leave visible pending work. Model aliases are recorded without claiming an immutable weights revision. `COLONYMIND_SOURCE_CLAIMS=off` stops the shared source projection worker; explicit erasure still attempts physical media cleanup.

A rejected final caption records `missing_final_answer`, `incomplete_final_answer`, `description_word_limit` or `description_character_limit`, with the returned model and routing provenance when a response exists. The limits remain 160 words and 2,400 characters. Rejected prose and reasoning never become descriptions or search entries. Other failures retain only their exception class. Status describes the latest attempt, not a complete attempt history; older generic `ValueError` rows cannot reveal which check failed. The existing retry schedule and original-image access continue independently of caption success.

Configure the named role through the existing host LLM configuration, for example:

```json
{"provider":"local","baseUrl":"http://127.0.0.1:8080/v1","models":{"vision":{"model":"local-vision-model","supportsVision":true}}}
```

`supports_vision` is also accepted. The declaration must follow an actual image qualification of that endpoint/model; it is not inferred from the name. Normal text scoring and escalation never select the VISION role.

A source deletion atomically removes its image links. If no surviving source owns the pixels, the description and search projection are removed and the original/thumbnail become cleanup work. Successful deletion removes those files; failure reports `media_cleanup: pending`, and later worker passes or an explicit retry attempt cleanup again. Shared pixels remain only for surviving sources, and the deleted contact cannot read or recall another contact's copy. Late caption results check lease ownership and surviving source links before committing.

Startup recovers files stranded before a source transaction committed, only in the source-owned namespace and under the ingest write lock. Backups of the ledger must include the image namespace; restoring the database alone cannot recreate original pixels. This increment does not erase external transport caches or old backups, and cannot retract evidence already sent in an earlier response.

## Recall and inspection

Descriptions have a rebuildable local FTS index. Authorized matches enter the same candidate selection, reranking and character budget as textual source evidence. There is no second media injection. The packet carries the asset handle, source turn, role, model alias and uncertainty label. Description recall remains available after changing the interaction model because descriptions and originals live outside the model.

The source status endpoint now includes recent media jobs:

```
GET /v1/host/memory/sources/claims/status?contact_id=...
```

Original bytes are available only through the authenticated, contact-scoped API, with source/session ownership checked and caching disabled:

```
GET /v1/host/memory/sources/assets/<sha256>?contact_id=...&session_id=...
```

There is no public static file route, and no Hermes tool opens an image since 1.10 (the adapter's source-read
tool went with its other native tools): the model sees a recalled image through its description. A client with the
`read` access opens the original through the scoped `POST /v1/host/memory/read` route with `source_view: "image"`,
a `source_id`, its `source_version` and the `asset_hash` (the 64 hexadecimal characters after `sha256:`), in the
participant's own scope and session. The route accepts no file path, URL, participant override or image service.
Opening another person's source or an asset not in that exact source fails.

Recalled media places its checked canonical `source_id` and `source_version`
beside the description. Use that source ID; the `media:` row ID and
`sha256:` asset ID identify different things. This metadata uses the existing
recall character budget. It never converts a media ID into a source ID.

The route returns original pixels together with their source
message role, recorded/reported times, exact source references and attributed
corrections applying to the image's owning messages. Interpretations remain
unverified. Corrections are not truncated to make an image fit: a correction
bundle over 16,384 characters returns unavailable without sending pixels;
the textual source reader can still page its evidence. Original static-image ingest limits still apply. Audio
and unsupported attachment types cannot enter this image view.

A read carries its read revision: a new applicable correction changes it even without erasure. A client applies it
before using a delayed result, and after erasure, changed ownership, a missing original or a changed correction set
it opens the source again rather than replaying the old pixels as fresh evidence. Evidence already sent cannot be
retracted. A data URL retained in a transcript, or a recalled description alone, is not evidence that a vision model
inspected the original.

## Qualification

The source tests exercise actual image files, source API ingestion, scope isolation, durable reopening, canonical hash preservation, checkpoint/outbox erasure agreement, late-description rejection and cleanup of originals/thumbnails. The native packaged adapter test uses Hermes' actual `build_native_content_parts` and lifecycle hooks, then checks the durable outbox and client serializer preserve the content list.

The source-read tests (`sidecar/tests/test_source_image_read.py`) reopen original pixels only through their current
source revision: a correction changes the revision without erasure and erasure revokes the pixels, an asset cannot be
borrowed from another owned source, a corrupt or oversized original never becomes image output, an attribution change
revokes an existing revision, oversized corrections never silently disappear, an erasure racing a file read never
publishes the pixels it caught, and an audio asset is never an image read. No inference calls are made. This
establishes protocol and lifecycle behavior, not visual accuracy or useful ordinary-conversation recollection.

One private neutral image was also described through an explicitly configured local VISION role in 2.47 seconds. The real source API, caption worker and context route correctly recalled the description in another session, withheld it from another contact, reopened the exact original bytes and erased the original/description. This is one working image loop, not a vision accuracy benchmark or production acceptance. Deployment still needs a real channel attachment observed through the deployed adapter and memory provider, followed by a useful visual question whose answer requires the original pixels. Future image embeddings should preserve the same source ownership and erasure lineage.
