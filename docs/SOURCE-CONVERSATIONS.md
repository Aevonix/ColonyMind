# Observed conversations

A wearable or room microphone can hear conversations around the owner that nobody addressed to the agent. A
capturing device (or the hub that transcribes it) sends each one, once and complete, to
`PUT /v2/host/conversations/observed/{conversation_id}`. The canonical source ledger keeps it word for word,
with its speaker labels, and with no expiry. It is not a turn: it is never filed as something the owner told the
agent, and only the words the device labelled `owner` are the owner's.

## The request

```json
{
  "source": "pendant",
  "captured_at": "2026-10-12T14:03:00Z",
  "ended_at": "2026-10-12T14:41:00Z",
  "recognizer": {"model": "asr-model", "revision": null},
  "speakers": [{"label": "owner", "contact_id": null}, {"label": "speaker_1", "contact_id": null}],
  "segments": [{"speaker": "speaker_1", "start_ms": 0, "end_ms": 1800, "text": "I live in Denver."},
               {"speaker": "owner", "start_ms": 2100, "end_ms": 2600, "text": "Nice."}],
  "summary": null,
  "action_items": []
}
```

- `conversation_id` matches `^[A-Za-z0-9][A-Za-z0-9_.:-]*$`, at most 128 characters (for example
  `device:<id>:<start_ms>`); slash is refused with `422` `invalid_conversation_id`. `source` names the capturing
  device kind (`^[a-z0-9][a-z0-9_.-]{0,31}$`).
- `captured_at` and `ended_at` carry a timezone; `captured_at` is not after `ended_at`, and `ended_at` is not in
  the future (two minutes of device clock skew are allowed). A conversation spans at most 24 hours.
- Segment offsets are milliseconds from `captured_at`, ordered by `start_ms`, each ending by `ended_at` (five
  seconds of rounding allowed). Every segment names a listed speaker and carries words.
- The label `owner` means the device verified the owner's voice. It is trusted only from a credential granted
  the owner, the only kind that reaches the route (`403` `not_owner` otherwise), and names no contact. Any other
  label is an unresolved speaker unless it names an existing contact (`422` `unknown_contact`); it never names
  the owner, shared or global person.
- `summary` and `action_items` are the device's own derivations. They are kept and recalled labelled as derived
  and unverified, never as transcript words, and an action item never becomes a commitment.
- Bounds (builder's choices, within the turn envelope's 8 MiB source limit): at most 10000 segments and 600000
  characters of segment text (`413` `observed_conversation_too_large`), 20000 characters a segment, 32 speakers,
  a 4000-character summary and 50 action items of 500 characters each (`422`).

Idempotency is keyed by the pair (`source`, `conversation_id`), so different sources may use the same id.
It is the turn route's: `201` and `Idempotency-Status: created` the first time, `200` `replayed` for an
identical replay, `409` `conversation_id_content_conflict` for changed content under the same source and id
(never a silent rewrite), `202` `in_progress` with `Retry-After` while the same request is being taken in,
`200` `erased` for a historically erased source, and `503` with `Retry-After` while the sealed store is unavailable.
Curated forgetting retains the source, so its identical replay remains `replayed` and stays fenced from recall.
Every write is the ledger's own
idempotent source write, so a request that failed part way is taken in again as new.
`GET /v2/host/conversations/observed/{conversation_id}?source=<source>` requires the device source and
returns the conversation as memory holds it now, to the owner only; `segments_complete` is false when an erasure
of exact messages removed part of it.

The device source is in the PUT body and the GET query, not the path. Capture checks the body's `source`
against the credential's grant, and a read checks the stored conversation's (`403` `source_not_granted` on
capture). A read of a conversation from a source the credential is not granted, of a historically erased one and of an
unknown id all answer
`404` `observed_conversation_not_found`, so a read never says whether an id was used.

## Storage

The conversation is one source of the owner's, `observed-conversation:<source>:<conversation_id>`, in its own
session and in person scope. It uses the existing ledger, secret sealing, lexical and semantic indexes, erasure
watermark, backup and orphan cleanup. Each message holds one window of consecutive segments small enough to be
a single recall excerpt, so every recalled excerpt opens with its header (the conversation, the capturing
device, its times, that it was not said to the agent and that the transcript is unverified) and every line
with its speaker label (`[owner]`, `[speaker_1]`, `[speaker_1, contact <id>]`). A long segment is split across
lines; the `GET` rebuilds it exactly. Only the ledger writes such a message: a turn that carries a block of
the same shape keeps it unretained.

Every speaker's words, including a resolved contact's, stay verbatim in this one owner source. They are
recallable owner-side only and never surface in that contact's chats. No per-speaker sources are written;
`speaker_source_ids` stays an empty list in PUT and GET responses. Turn-writing routes refuse IDs starting
with `observed-conversation:` (`422` `reserved_turn_id`), reserving the namespace for this writer.

## Recall

Lexical and semantic recall find a conversation like any source. Its excerpt is a `source_quote` with role
`observed_conversation`, state `derived_unverified`, source modality `observed_conversation` and an
`observed_conversation` handle: the conversation id, the device, `captured_at` and `ended_at`, the speaker
labels, the namespaced `source_id` and `said_to_agent: false`. Its `reported_at` is the capture time. It is never
paired with a reply.

## Claims and commitments

Observed source-claim jobs are backlog work under `projections.backlog_per_hour`. Live turns go first;
each observed window spends one budget admission and then yields. A transient failure resumes at the first
unfinished window, including when earlier windows formed no claims.

The existing source-claim job forms owner claims from each window, using the [audio transcript](SOURCE-AUDIO.md)
path's segment grounding. The extraction view of a window adds the end of what was said just before it, and
only the `owner` lines are quotable. The other lines are supplied as context to interpret answers, agreement and
quotations. A quotation outside the attributed speaker's words is refused deterministically, before review
(`audio_segment_grounding`), whatever the model proposes: `speaker_1: I live in Denver.` beside `owner: Nice.`
never becomes a claim about the owner. No claims about contacts are formed from observed conversations yet.
Every other speaker's words stay retained and recallable owner-side only and form no claim.

An admitted claim is `derived_unverified`, with source modality `observed_conversation` and an evidence basis
naming the conversation, the segment, the speaker, the recognizer and the provenance
`reported_in_observed_conversation`. The conversation's capture time anchors relative dates.

Commitment capture reads only the owner's own lines of the owner's source, marked as heard in a conversation
with other people and not said to the assistant. It uses only the last 6000 characters of the combined
owner-text view; earlier owner words remain stored and recallable but are outside commitment extraction.
Other speakers' words and the device's action items form no commitment. Appraisals and opinions do not read
observed conversations.

## Erasure

`POST /v1/host/memory/curated/sources/forget` with `person_id` and the conversation's source id in `source_ids`
appends a curated tombstone. Recall and curated retrieval suppress the source and recorded dependencies;
source conversations, claims, vectors and commitments remain in historical storage. Repeating the same
selection returns the original durable receipt. There is no destructive cleanup or source-row deletion.
The authorized raw conversation reader remains an inspection surface. Historical per-speaker sources from
older writers are also fenced from recall while their rows remain retained.

## Limits

This keeps what a device transcribed; it does not verify the recognizer, the speaker labels or the owner's
voice beyond trusting the owner's credential. It performs no audio capture, keeps no audio, and identifies no
speaker from sound. A claim spanning two lines, or needing more context than its window, stays evidence for
recall rather than an assertion.
