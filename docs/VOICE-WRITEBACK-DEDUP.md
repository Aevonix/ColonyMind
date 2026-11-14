# Staged voice write-back dedup

The original voice producer owns canonical ColonyMind capture. The write-back
plugin appends normal Hermes user/assistant rows without invoking lifecycle
capture. Those rows carry `display_metadata.bridge = voice.writeback.v1`,
the stable exchange ID/digest, and a confirmed original `source_receipt`.

`checkpoint()` filters this provenance before reducing messages to role/content
for the durable compression outbox. The memory provider already delegates its
pre-compression checkpoint to this helper. Its session-end fallback also skips
a last exchange containing a confirmed imported voice row. A new text turn
after the import is still captured normally. Original voice sources/outbox
entries are never modified, deleted or delivered again by this filter.

Textual voice markers alone, missing/false source receipts, and incomplete
playback markers do not qualify. This relies on trusted Hermes row metadata
from the owner-authenticated producer/bridge; it does not infer capture from
text similarity or perform a network source lookup. Deploy the paired bridge
and this adapter only after separately authorized live integration checks.
