# ColonyMind memory provider for Hermes

`colonymind-memory` fills Hermes' external memory-provider slot with ColonyMind. For every message it assembles
recall for that message, before the reply, scoped to the person speaking: their own words with speaker, time and
source, the commitments and state that concern them, and the current time. It ships in the `colonymind-hermes`
package beside the [`colonymind` plugin](../hermes-plugin/README.md), which captures turns and runs the mind.

Hermes runs its built-in memory plus at most one external provider, so selecting this one replaces any other.
`colonymind init` names the provider it replaces and `colonymind init --uninstall` puts it back.

## Setup

Either front door runs the same `colonymind init`:

```bash
uv tool install --python 3.12 colonymind && colonymind init && hermes gateway restart
```

or, with this package already in Hermes' environment, `hermes memory setup`, then `colonymind-memory`. The setup form
asks only for the connection (the sidecar URL and the key file); it then runs `colonymind init` for the instance that
key file belongs to, offering first to install the sidecar as a tool of its own when it is missing (never into
Hermes' environment).

The provider reads the shared keys `colonymind init` writes (`plugins.colonymind.sidecar_url` and `key_file`), with
`memory.config` and the profile's `colonymind-memory.json` as overrides:

| Key | What it sets | Default |
| --- | --- | --- |
| `url` | the sidecar URL | `plugins.colonymind.sidecar_url`, else `http://127.0.0.1:7777` |
| `key_file` | the file holding the sidecar's key (or `COLONYMIND_API_KEY`) | `plugins.colonymind.key_file` |
| `contact_id` | the owner's contact, when it is not the instance's `owner.contact_id` | the instance's owner |
| `timezone` | the clock's time zone | UTC |

The provider reads memory context. Enable the general `colonymind` plugin in each selected profile to capture
completed turns and final session exchanges; provider-only profiles do not capture them.

Recall notices use two keys in Hermes' config:

| Key | What it sets | Default |
| --- | --- | --- |
| `plugins.colonymind.recall_notice.destination` | `origin` keeps Hermes' status line; `home` sends to the turn platform's configured home chat; `off` disables notices; `<platform>:<chat_id>` sends to that chat | `origin` |
| `plugins.colonymind.recall_notice.details` | `true` adds short, redacted memory previews with dates to routed notices; other people's memories appear only as an aggregate count | `false` |

For `home`, the provider passes the turn's platform to Hermes, which resolves its home chat. CLI and TUI turns
fall back to `origin`. An unknown destination also falls back to `origin` and is logged once. The origin status
line stays unchanged and outside the model's context. Routed notices start background delivery only when Hermes
calls `recall_status`, never during prefetch. Each process allows one outstanding send; notices arriving while it
is busy are skipped and logged. A send lasting over 10 seconds logs a warning and keeps its slot until it returns.
Hermes may mirror routed notices into the destination chat's history.

The provider reports itself unavailable, with the reason and the fix, when the URL or the key file is missing or
unreadable. That check reads configuration only, never the network,
and a sidecar outage never switches memory off for a session.

## Each turn

- **Recall.** One bounded context assembly per turn, keyed on the message being answered; nothing is prepared in the
  background for a message not yet known. The whole prefetch ends within Hermes' own wait. The context is fitted to
  what Hermes puts inline, and a section the session's last turn already carried unchanged becomes one line, so the
  cached prompt prefix holds.
- **Scope.** A turn is bound to its sender. The owner's turns get the owner's context; a contact's turn gets only
  that contact's own evidence, claims and media, the commitments shared with them and their digest, never the
  owner's graph, tasks or anything global. A real channel with no bound sender gets no context. Failure or timeout
  yields no ColonyMind content, never someone else's.
- **Indicator.** By default, in the owner's own chat Hermes shows "recalled N" for the quotations the turn received.
  Notices can be routed or disabled with the keys above. Only the owner's private turns produce notices; a contact's
  turn or a group never does.
- **Tools.** The owner's lane gets one write, `colonymind_resolve_commitment`. A guest or unbound session is offered
  no tool. Explicit memory search is the `colonymind` plugin's `colonymind_memory_search`.
- **Forgetting.** To forget something, name its sources: the owner's `colonymind_memory_forget` tool, or
  `POST /v1/host/memory/curated/sources/forget`, with the `source_id` of each source a search (`colonymind_memory_search`)
  or `colonymind memory inventory <who>` shows. Send `person_id` and `source_ids`. The answer includes
  `source_erased: true` for compatibility, meaning a curated tombstone was recorded; historical data is retained.
  A durable `receipt` records the operation. Repeating the same ids returns the same receipt. Text/voice recall,
  curated reads/search and allowance repacking suppress the forgotten evidence and its recorded dependencies.
- **Hermes' memory files.** Editing them never creates a statement in ColonyMind, and removing an entry is not a
  reliable way to forget. When the agent's memory tool removes one, the provider sends the removed text with the
  current session, and ColonyMind forgets a source only when that text is a whole message of exactly one source of
  this session. An entry the model wrote in its own words or as a summary, a message from an earlier session, or
  text that more than one source of this session holds (a message and its copy in a compression checkpoint are
  two) forgets nothing and stays recallable; a change made to the file by hand never reaches ColonyMind. The provider's
  diagnostics give the last result (`source_erasure`).
- **Compression.** Before Hermes compresses a session, the direct conversation is checkpointed to the source ledger,
  bound to the turn's participant, so compressed turns stay recallable.
- **Backup.** `hermes backup` includes the ColonyMind instance directory.

## Outages

A sidecar outage keeps the provider attached so later turns reconnect. After three failed requests a circuit
breaker skips the sidecar for 60 seconds. `hermes colonymind-memory status` reports health, and the provider's
diagnostics say whether the connection is `unverified`, `connected` or `degraded`.

See [the Hermes adapter guide](../../docs/HERMES-ADAPTER.md#memory-provider) and
[recall](../../docs/RECALL-HYBRID.md) for the details.
