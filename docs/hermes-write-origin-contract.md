# External write origin contract

External MCP tools execute in a separate process. The gateway's ColonyMind
adapter registers each selected tool call with the sidecar before Hermes
executes it. The MCP server claims the record through its own narrow credential.
Hermes core and the external MCP server are separate from this implementation.

An owner message on any owner channel, or an autonomy run with owner standing,
starts a turn without a confirmation prompt. A send or calendar invitation needs
an owner action when its destination appeared solely in content read during the
turn: exclude destinations in the owner's own message, known contacts, and
existing participants in the destination thread. Any external write from another
origin needs an owner action. The MCP consumer identifies write operations and
existing thread participants and fails closed when no origin record is available.

## Gateway registration and MCP names

By default only MCP server `google` participates. Configure raw server names in
Hermes config:

```yaml
plugins:
  colonymind:
    tool_origin_servers: [google]
```

An empty list disables registration. Built-in tools, other servers and unregistered
lookalike names do not participate. The guard registers only calls its existing
rules allow to proceed.

In the supported Hermes source, `tools/mcp_tool_registration.py` registers native
tools through `mcp_prefixed_tool_name` in `tools/mcp_tool_schema.py`:
`mcp__<sanitized server>__<sanitized tool>`. Each character outside
`[A-Za-z0-9_]` becomes `_`. If the full name exceeds 64 characters, Hermes
uses its first 55 characters followed by `_` and the first eight hex digits of
SHA-256 of the full prefixed name. The agent passes that registry name into
`pre_tool_call`. Each registered native handler closes over the original raw
`server_name` and `tool_name`; the adapter checks its `mcp-<raw server>`
toolset and uses these original names. For ordinary names this removes
`mcp__<server>__`; for sanitized or shortened names it also recovers the exact
original bare tool, which cannot be reconstructed by splitting the visible name.

The adapter uses its existing instance credential (`api.key`) to call:

```text
POST /v1/mind/tool-origins
{tool, args_hash, origin, channel, turn_id, session_id, run_id?, untrusted_addresses}
```

`tool` is the original bare MCP tool name. The hash is SHA-256 of UTF-8
`json.dumps({"tool": bare_name, "args": args}, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)`. All arguments participate, including
bodies; non-finite numbers are rejected. The POST timeout is 0.5 seconds.
Registration errors log one line containing only the exception type and block
dispatch. This prevents an MCP execution from claiming a stale record when the
sidecar is unreachable. A missing record causes the MCP consumer to fail closed.

`origin` is `owner_channel`, `autonomy`, or `other`. The guard's resolved
owner sender and evidence that this exact turn began with their inbound message
establish `owner_channel`, including the current owner speaker in a group. An
installed hands bridge must attest `owner_started_turn`; otherwise the gateway's
own ingress record must bind the triggering message. Internal, system, notification,
cron-delivery and relay turns carry `other`. An owner creation record in `StandingStore` establishes autonomy
for a kanban task or cron job. Only autonomy carries `run_id`. Inherited owner
children, unknown standing, arguments, model text and headers do not establish
owner origin. Unknown and non-owner calls carry `other`.

The adapter collects lowercase email addresses from this session and turn's
tool results, subtracting the owner's own words and known contacts. Contact
lookups are cached per turn, with a 0.2-second per-address timeout and a one-second
total lookup budget. Failure treats an address as untrusted. It retains at most
128 turns and 256 result addresses per turn, discarding them at native completion,
failure or interruption. A 257th distinct address marks the turn as overflowed:
`untrusted_addresses` then returns only `*`, and the registration and claim carry
`["*"]`. The MCP consumer must treat **every recipient as untrusted** when `*`
is present, requiring an owner action even for known contacts or addresses in the
owner's message. The MCP consumer must also apply its own recipient limits.

## Sidecar claim and credential

Records live only in sidecar memory for 120 seconds, keyed by `(tool, args_hash)`.
The store holds at most 4,096 records and evicts the oldest at capacity.
Registration replaces an existing record with the same key. Since stdio MCP calls
cannot carry a per-invocation nonce, two live records that disagree about origin,
session, turn or recipient provenance downgrade the newest record to `other` with
`untrusted_addresses: ["*"]` and no autonomy run id. This downgrade persists until
claim or expiry. A rejected registration with a valid key invalidates the prior
record. Claims remove a record atomically, including under concurrent requests:

```text
POST /v1/mind/tool-origins/claim
{tool, args_hash}
-> {origin, channel, turn_id, run_id?, verified: true, untrusted_addresses}
```

An unknown, expired or previously claimed key returns 404. Registration requires
the adapter's instance credential. Claim and owner-actions require an issued
credential with `access: [tool-origin]`, no owner platforms and `contacts: own`.
This grant stands alone: no manage, read, write, answer, observe or legacy scopes.
It reaches only these two POST routes; it cannot register origins, read memory,
read permit asks, answer asks or access the vault. Other credentials, including
owner/manage credentials and the adapter, cannot claim or request owner actions.

Mint the MCP process's credential through the existing CLI flow (example only):

```sh
colonymind credentials --home /path/to/instance add google-tools --access tool-origin --contacts own --key-file /path/to/private/google-tools.key
```

Do not add an owner platform. Supply the resulting secret only to the MCP process
and restart the sidecar after updating its credential table, as the CLI instructs.
No credential is minted by this build.

## Owner actions

```text
POST /v1/mind/owner-actions
{tool, args_hash, summary: {recipients, subject_or_title}, reason}
-> {state: asked|approved|denied, ask_id, owner_action?: {ask_id, args_hash, settled_at}}
```

The sidecar reuses the existing `Mind._hold(origin="hand")` creation path.
Terms contain the tool, recipient emails and a secret-redacted subject or title.
Bodies, attachments, headers and the supplied free-form reason do not enter the
ask ledger. The full argument hash binds the ask as its effect digest.
The ask is a non-executing note and offers the existing yes/no choices.

Repeats for the same credential, tool and hash reuse their ask. The sidecar reads
the existing permit-ask view and verifies its id, digest, approved state and the
recorded owner's yes. A yes yields one receipt within ten minutes of settlement;
a no, expired or future settlement, closed ask, changed binding or consumed yes
denies. Creation, polling and consumption share a lock. Different argument hashes
get separate asks. Up to 1,024 bindings per sidecar process are retained; new
bindings deny at capacity. A restart discards local bindings and must obtain a new
ask and owner yes rather than reconstructing approval from local receipts.

## Vault fill confirmation

The existing vault-captures route also accepts the Companion phase-two payload:

```json
{"kind":"vault_fill_confirm","origin":"https://example.invalid",
 "site_hint":"example.invalid","label":"Example login","ttl_s":120,
 "save_toggle_default":false,"fields":[]}
```

`site_hint` is optional. Origin follows the existing HTTPS-origin validation.
The item label contains no secret and is at most 120 characters. TTL is an integer
from 1 through 120, the save toggle must be false, fields must be empty, and extra
keys are rejected. The owner sees “Fill your saved login for Example login on
https://example.invalid?” with the existing yes/no choices. Existing typed vault
kinds retain their schemas and behavior.
