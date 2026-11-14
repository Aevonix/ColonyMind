# MCP: a read-first endpoint for other agents

An agent that speaks the Model Context Protocol (a coding agent, a desktop assistant, an editor) can read what
ColonyMind holds: memory, the open commitments, the mind's status, why it did something and what it is asking. The
sidecar serves it at `POST /v1/mcp`, over the protocol's streamable HTTP transport.

Two things hold everywhere:

- **Bound to a credential.** Every request carries a credential issued for that client. The credential alone decides
  whose memory the tools read and which tools the client gets; no tool takes a person.
- **Read first.** Five tools read. The sixth, `note`, leaves a note under the credential's own identity, where it is
  evidence and never anyone's claim. Nothing answers an ask, changes a grant or the autonomy level, or erases, and no
  model sits behind the endpoint.

The Hermes adapter does not use it: Hermes has the native plugin and memory provider
([Hermes adapter](HERMES-ADAPTER.md)). What the endpoint adds to another host's agent is unmeasured.

## Set it up

1. Issue the client a credential of its own. For a client that works for the owner:

   ```bash
   colonymind credentials add coder --owner-platform mcp --access read --access note \
       --key-file ~/.config/coder/colonymind.key
   colonymind service restart
   ```

   `--owner-platform` is what gives a credential the owner's view. With `--access read --access note` it writes no
   turn at all, so the platform name (`mcp` here) only marks the grant. Leave out `--access note` for a client that
   only reads. The instance key (`api.key`) is the Hermes adapter's and is refused here: a client with a credential of
   its own can be narrowed or retired without touching Hermes.

   A client that works for one other person is bound to that person in `colonymind.yaml`
   ([Credentials](CREDENTIALS.md)):

   ```yaml
   credentials:
     front-desk:
       sha256: "<digest of its secret>"
       owner: []
       contacts: [cid-front-desk]
       viewer: cid-front-desk
       access: [read, note]
   ```

2. Register the endpoint in the client, with the secret as a bearer token. Most clients take a URL and headers:

   ```json
   {
     "mcpServers": {
       "colonymind-mcp": {
         "type": "http",
         "url": "http://127.0.0.1:7777/v1/mcp",
         "headers": {"Authorization": "Bearer <the secret>"}
       }
     }
   }
   ```

   Any name works. `colonymind` is the name the retired stdio server was registered under, and `colonymind upgrade`
   still names an entry called that in another application's configuration as the retired server; another name
   keeps its report quiet.

The sidecar listens on loopback by default (`sidecar.host`, `sidecar.port` in `colonymind.yaml`). The secret travels
as a bearer token over plain HTTP, so a client on another machine reaches the endpoint through an SSH tunnel or a
private network, not an open port.

## Who the tools act for

| The credential | The tools act for | The mind's view |
|---|---|---|
| granted the owner (`owner` lists a platform) and bound to no list of persons | the owner | the owner's |
| bound to a list of persons (`contacts: [...]`) | its `viewer`; without one every person-bound tool is refused (`person_binding_required`) | that person's |
| anything else (`contacts: own`, or `contacts: all` without the owner) | its own identity, `credential:<name>` | a non-owner's |

An argument that names someone (`contact_id`, `person_id`, `viewer`, anything the schema does not list) is refused;
it never changes who the tool reads for. Each read calls the route that serves the same read, with the credential's
own authority, so the route decides the view as it does for any caller: the owner's view only for a credential
granted the owner, a guest's for the rest. On the API the mind's and the commitments' routes need `manage`, which
also reaches every change; here their reads need only `read`, so a client can see the owner's view without holding a
credential that could change anything.

## The tools

| Tool | Access | Arguments | What it returns |
|---|---|---|---|
| `recall` | `read` | `query` (up to 4,096 characters), `limit` (1 to 20, default 5), `explain` | What memory holds for the person, with its sources: the memory search (`POST /v1/host/memory/search`). `explain` adds why each candidate was kept or left out, as `colonymind recall --explain` prints it. |
| `open_commitments` | `read` | `limit` (1 to 200, default 50) | The pending and overdue commitments, overdue first: every one for the owner; for anyone else only those their own words already show, by the rule their conversation context uses, up to the limit (the context itself lists five at most). |
| `mind_status` | `read` | none | Whether the mind is on and its autonomy level; the owner's view adds what it is doing, its open asks, drives, budgets and faculties. |
| `why` | `read` | `intention_id` | One intention's audit entry: what the mind did or asked, why, and how it ended. Another person's entry is not found. |
| `asks` | `read` | none | The open asks, each with its code, reason and expiry; none outside the owner's view. Listing only: the owner answers in their own chat ([Mind](MIND.md), Asks). |
| `note` | `note` or `write` | `text` (up to 8,000 characters) | The note is kept as a turn written as `credential:<name>`, in the session `mcp:<name>`; the result names its `source_id`. |

A note is session evidence ([Credentials](CREDENTIALS.md#a-credentials-own-identity)). It is never projected into
claims, appraisals, commitments or opinions and never raises a question to the owner, whoever the credential speaks
for: a client granted the owner's view still cannot put words in the owner's memory. A credential bound to its own
identity reads its notes back with `recall`. The owner forgets a note like any other source:
`POST /v1/host/memory/curated/sources/forget` with `person_id: credential:<name>` and `source_ids` containing
the source id the note returned. A tombstone suppresses recall while retaining the historical note.

A tool the credential's access does not include is not listed, and a call to it is refused. A credential whose access
includes neither `read` nor `note` (nor `write`) gets no tools and is refused at the door; so is one carried from
1.9.0, whose scopes name no tool.

## The protocol

- Every request is a `POST` with one JSON-RPC message, or a batch of them (the 2025-03-26 transport), and gets a JSON
  answer. There is no server-sent stream and no session: a `GET` or `DELETE` gets 405, and every request carries its
  credential.
- `initialize` answers with the client's protocol version when it is 2025-11-25, 2025-06-18 or 2025-03-26, else with
  2025-11-25. A later request that names another version in `MCP-Protocol-Version` gets 400.
- Notifications and responses alone get 202 with no body. The methods are `initialize`, `ping`, `tools/list` and
  `tools/call`; any other method is `-32601`, and an unknown tool `-32602`.
- A tool that cannot answer (arguments that do not fit its schema, the mind not running, an entry not found, a missing
  person binding) returns its reason as a tool error (`isError: true`), so the client's model can correct the call.
- Each result carries its text (the tool's own rendering, or its JSON) and the same value as `structuredContent`.

## Refusals

| Status | `detail.code` | When |
|---|---|---|
| 401 | `credential_required` | no credential, a secret that matches none, a secret whose time is up, or an instance with no key (development mode serves nothing here) |
| 403 | `instance_key_refused` | the instance key, or one of its other secrets: issue the client a credential |
| 403 | `mcp_not_granted` | the credential's access includes none of `read`, `note` and `write` |
| 403 | `origin_not_allowed` | the request carries an `Origin` that is not a loopback one: a page that rebinds a name to this machine sends its own origin |
| 400 | (JSON-RPC `-32600`) | an `MCP-Protocol-Version` this endpoint does not speak, or a malformed message (`-32700` for a body that is not JSON) |
