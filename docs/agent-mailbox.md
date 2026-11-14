# Receive-only agent mailbox contract

The agent has its own mailbox on the deployment's mail domain. No sending path
exists, including test-send, replies, drafts, forwarding, BCC, automatic
receipts or sender-verification email.

## Ownership

The mailbox is implemented in **ColonyMind**, inside the existing sidecar. Its
[architecture](ARCHITECTURE.md) assigns durable identity and retained data to the
private ColonyMind instance, and transport translation to adapters. This module
owns only identity and retained inbound data; a later Hermes integration owns
IMAP/provider lifecycle. No Hermes core changes, new service, scheduler or flag.

## Data and trust boundary

Mailbox identity: `mailbox_id`, immutable `address`, `display_name`, `status`.
Address local part must match `[a-z0-9][a-z0-9-]{1,30}[a-z0-9]` exactly;
domain is the instance's `COLONYMIND_AGENT_MAIL_DOMAIN` setting. No case
conversion, whitespace trimming or alternate domain silently repairs an invalid
address. Registering a local identity never reserves a provider address. Default
status is `unprovisioned`; `active` is an operator assertion that external setup
is complete; `disabled` stops ingestion. No identity is installed by default.

The mail domain is the deployment's own value and has no default. Set it among
the sidecar settings of the instance's `colonymind.yaml`
([configure](INSTALL.md#configure)) and restart the sidecar:

```yaml
environment:
  COLONYMIND_AGENT_MAIL_DOMAIN: example.com
```

It is read, in lower case, each time an identity is registered. Without it,
registering is refused with `mail_domain_not_configured` and nothing is stored.
An identity keeps the address it was registered with, and keeps receiving, when
the setting later changes or is removed.

Inbound message: stable local `message_id`, mailbox-scoped `delivery_id`,
`thread_id`, sender, recipient list, subject, plain `text`, occurrence time,
server acceptance time, nullable `in_reply_to`, and local read state. A trusted
adapter supplies delivery/thread identity; neither a body nor a From header
authorizes a caller or establishes a trusted contact. Retries of an identical
delivery return the existing record; conflicting reuse is rejected. Deleted
thread deliveries remain deduplicated without retaining their content.

All sender/subject/summary/body strings are **untrusted data, never model
instructions**. Every thread/message projection carries
`content_trust: "untrusted"`. There is no model call, memory ingestion, tool
dispatch, sender trust change or automatic action on receipt/read. Companion
renders plain text, without HTML execution, external images, automatic links or
attachment execution. Summaries are bounded text excerpts, not model summaries.
HTML-only/attachment-only messages have empty text for the app's no-text fallback.
Future model consumers must retain this boundary and require an explicit owner
request; mailbox content cannot change system/developer instructions or grants.

## Companion API

Storage is `agent-mailbox.db` under the selected private instance state
directory, with a mode-0600 database, SQLite WAL, a busy timeout and per-operation
connections. One store is created lazily per instance. Schema initialization or
migration runs only when the schema version needs it; ordinary reads use a
consistent transaction snapshot without a writer lock. Paging, bounded excerpts,
counts and aggregates run in SQL. Receive, read and delete commit atomically with
their delivery tombstones and unread projections. Delete uses `secure_delete`
and a truncating WAL checkpoint before returning success. A concurrent read
snapshot may delay that purge: 503 means retry Delete to complete it, even if
the logical deletion has already committed. No startup worker is needed.
Corrupt storage fails closed rather than silently starting empty. Back up the
database with SQLite's backup mechanism or with existing
private state while closed; physical secure erase of backups is separate.

All routes require an authenticated issued Companion owner relay credential,
with the named `companion` owner platform and `manage` access, matching the
Companion device and owner route boundary. The Hermes/gateway instance key,
its rotation secrets, all-platform model credentials, anonymous, guest and
legacy-scoped credentials are denied. No body/query field can grant owner access.
Companion uses its existing authenticated owner backend relay. Successful
responses use `Cache-Control: no-store`. IDs are opaque; absent or cross-mailbox
resources return 404.

The following paths are relative to `/v1/agent-mail`:

| Method / path | Request / result |
| --- | --- |
| `GET /mailboxes` | `{mailboxes: [identity]}`; empty array means setup required |
| `POST /mailboxes` | `{local_part, display_name}` → identity, 201; duplicate address 409; no mail domain configured 501; local record only |
| `GET /mailboxes/{mailbox_id}` | identity and global `unread_count` (messages), `unread_thread_count` |
| `PATCH /mailboxes/{mailbox_id}` | `{display_name?, status?}` → identity; address/id immutable |
| `GET /mailboxes/{mailbox_id}/threads?limit=50&cursor=…` | `{threads, next_cursor, has_more, unread_count, unread_thread_count}` |
| `GET /mailboxes/{mailbox_id}/threads/{thread_id}?limit=50&cursor=…` | `{thread, messages, next_cursor, has_more}`; does not mark read |
| `POST /mailboxes/{mailbox_id}/threads/{thread_id}/read` | `{through_message_id}` → read receipt and global unread counts; only messages through the displayed message become read |
| `DELETE /mailboxes/{mailbox_id}/threads/{thread_id}` | local deletion receipt and global unread counts; repeat deletion succeeds |

`limit` is 1–100. Inbox order is newest accepted delivery first; thread messages
are oldest accepted first. Cursor continuations retain a snapshot boundary, so
new deliveries do not move an existing thread into a second page or extend a
thread page unexpectedly. The app refreshes the first page to see new mail.
`next_cursor=null` / `has_more=false` means exhausted. Never increment an offset
locally; Load more passes the returned cursor to the same mailbox/thread route.
Malformed or cross-resource cursors fail with 422; cursors carry no authority.

Thread projection: `thread_id`, subject, latest sender, latest text excerpt
`summary`, `message_count`, latest `received_at`, `unread_count`, `unread`,
`content_trust`. Unread counts are current, even on snapshot continuation pages.
Preserve raw empty subject/text; Companion displays `(No subject)` and its
no-text fallback. A read receipt includes `read_count`. Delete purges message
content locally and returns `deleted` plus the thread ID; it does not expunge the
provider's mailbox. Retained delivery tombstones prevent replay resurrection.
An unseen later delivery may recreate the thread; replaying a deleted delivery
still reports it deleted and retains no old content.

No HTTP inbound webhook exists. `MailboxStore.receive(mailbox_id, InboundMessage(...))`
provides the in-process ingestion API for a future trusted adapter; no adapter
currently calls it. It accepts only active
identities, checks the envelope includes the mailbox address, bounds fields,
and persists before acknowledging. No credentials, raw MIME, headers or
attachments are accepted in this provider-neutral record. Transport authentication,
MIME decoding, provider UID/thread mapping and adapter activation remain future
work described in [provisioning](agent-mailbox-provisioning.md).

Errors: 401 missing/invalid authentication with a configured instance; 403
authenticated caller lacks Companion owner relay authority; 404 absent resource;
409 duplicate identity/conflicting delivery/inactive mailbox; 422 invalid
address, input, cursor or read marker; 501 no mail domain configured; 503
database busy or WAL purge blocked, with `Cache-Control: no-store`. Error
details contain codes, not mail text.

## Implemented and deferred

Implemented: inbox, thread, paging/load-more and unread projection; local
identity and display name only; local Delete only. Test-send, verification,
sending/contact rules, Trust, real address availability/provisioning and
provider mailbox deletion are deferred. Companion must omit those
actions for this receive-only slice; there is no capability flag or placeholder
route pretending they work. Local identity registration is never advertised as
successful provider provisioning.
