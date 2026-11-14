# Secrets

A secret that arrives in conversation (an API key, a password, a PIN, a one-time code, a card number) is not
dropped and not kept in the clear. It moves into an encrypted store, and a reference stands where it was:

    my key is [secret S-7KQ2M4XA: API key]

Memory keeps that the secret exists, what it is and when it came, so "the API key I gave you" still finds the
turn. The value itself never appears in memory search, in the transcripts the agent reads back, in the lexical
index or in any queue derived from a turn (claims, opinions, appraisals, commitments, vectors). The owner can
still use it: by reference, model-blind wherever possible (below).

## Which store

- **Configured credentials** (provider and tool keys) are the host's. With Hermes they live in Hermes' own
  configuration (`~/.hermes/.env` and its secret sources), and personal data sources come through Hermes
  skills and cron with their credentials there too; the sidecar has no connectors of its own. The
  connector-credential store ColonyMind used to carry (`SecretsManager`, its `env`, `keyring` and 1Password
  backends, the `/v1/host/secrets/*` routes and the `colonymind secrets get|set|delete|backend|status` verbs) is
  gone with the connectors that read it: it duplicated the host's store. The upgrade report names the backend
  that was configured and, for the file backend, the key names it held (never a value); those values stay where
  they are, for the owner to move into the host's configuration where a skill needs one.
- **Secrets that arrive in conversation** have no place in either host store: Hermes' vault holds only autofill
  items (a login needs a site origin), and its `.env` holds configured names, not references scoped to a
  person. `colonymind.secrets` is that store: sealed values, referenced from memory, each scoped to the person
  whose message carried it.
- `colonymind.redact` stays what it was: it masks what is shown and logged (phone numbers included), a different
  job from sealing what is stored.

## The detector

`sidecar/colonymind/secrets/detect.py`, the rules in the order they claim a span:

| Rule | Finds |
|---|---|
| `prefix` | vendor token prefixes with a minimum tail length (model and cloud API keys, code-host and chat tokens, model-hub, payment and mail keys) |
| `jwt` | three dot-separated segments whose first two start with `eyJ` |
| `private_key` | PEM and PGP private key blocks |
| `webhook` | chat webhook URLs |
| `url_userinfo` | the password in `scheme://user:REDACTED_CREDENTIAL@host` |
| `url_token` | a credential-named query parameter of 16 characters or more |
| `assign`, `json_field`, `bearer` | `NAME=value` where the name carries KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL or PRIVATE_KEY (a bare `API_KEY` or `TOKEN` included) and does not name something about one (`_ID`, `_URL`, `_TYPE`, `_TTL`, `_REF` and the like); the same names as JSON fields; a bearer token |
| `labeled` | a secret label (password, passcode, PIN, OTP, verification, security, backup or recovery code, recovery or seed phrase, API, secret, access or private key, auth or access token, CVV) followed by `is`, `=`, `:` or `->` and a value; a seed or recovery phrase is all of its twelve to twenty-four words |
| `telegram` | bot tokens |
| `card` | a card brand's prefix and length with a valid Luhn digit, written grouped or labeled as a card |

A value found by `assign`, `json_field`, `bearer` or `url_token` must look like a credential: 8 characters or
more, not a placeholder (masks, `${VAR}`, `<...>`, `your_...`, `example`, `redacted`, `changeme`, booleans,
`none`), not a stopword, and three character classes, or two and 16 characters, or 32 characters. A `labeled`
value is 4 to 12 digits (3 or 4 for a card's security code), or 6 characters of two classes, or 12 characters;
"password is wrong", a bare `token:` or `code:` never count. A quoted value is exactly the bytes between its
quotes; an unquoted one loses a sentence's closing period and a bracket that closes an aside.

**Never a secret:** a reference or a withheld marker, phone numbers, UUIDs, contact ids and long digit runs (order
and account numbers, millisecond timestamps), whatever names them; uppercase alphanumeric ids (tracking and
booking references) unless a secret's label calls one a password or a code; commit hashes and short hex ids in
text (a credential's name or label still seals a hex value). A reference is passed over only when its label is a
label: the vocabulary's characters and no value the detector finds, so an invented reference cannot hide a value.

**Calibration.** The rules were chosen by running candidate rules read-only over a real instance's stored turns
and stores, printing counts and character-class shapes only (never a value, a context string or an id), and
keeping only rules whose hits were credentials while every must-survive shape above survived. Two candidates were
rejected that way: a Luhn-only card rule (it matched contact ids and timestamps, never a card) and a generic
entropy rule (ids, URL paths, base64). A false positive costs one value in memory, and the owner can release it
(below). The fixtures in `sidecar/tests/test_secrets_detect.py` pin every rule's positive and every must-survive
shape; a rule can be turned off in configuration, and none is added without a fixture.

## Where secrets are sealed

- **Turn capture** (`/v1/host/turns/sync`, `PUT /v2/host/turns/{id}`): the envelope is sealed before its session
  is derived, before its digest and before the turn id is reserved; `_process_turn_sync` seals again at its top
  for direct callers. The summary, topics, communications log, preference learner, opt-out, the mind's owner
  turn and every event see references only. The scope is the sender's contact when the envelope names a sender
  (read, never created; an unknown sender's handle otherwise, never the contact the client claimed), else the
  envelope's contact.
- **The source ledger** (`TurnIdempotencyLedger.record_source`) seals every writer's messages (turn capture, the
  Hermes history import, judgments, outcomes, forecasts) before the digest, the row, the lexical
  index and every queue: every text a message carries, its content and metadata, structure excepted only where it is
  structure (a message's own role and source references; in its provenance, a native observation's identity and
  its arguments' digest; the same names deeper are words). A string held under a field name is read with that
  name too (`"password": "..."`), as the serialized text showed it. Sealing is idempotent (a reference is never
  detected) and deterministic (the same value in the same scope has the same reference), so a replay stores the
  same bytes and digests. Source annotations are sealed the same way, their request digest taken over the
  sealed words.
- **Staged, then stored.** A source's references are worked out without writing to the vault; its values are
  stored only once the ledger admits the source, just before its row (turn capture stores the rest of an
  envelope's values once the turn is admitted). A replay of an erased source, a conflicting one, or a message an
  erasure removes stores nothing, so a value the owner forgot stays forgotten. A forgotten reference is never
  revived: the same value arriving again later takes a new reference.
- **Erasure** is checked on every form of each incoming message, computed in memory and never stored: the sealed
  form, the form that arrived, the form its references resolve to (a gateway may have sealed the value before
  the sidecar ever saw it) and the form it had under each reference its values have had (before the owner
  released one, or before they forgot one and it came again under a new reference; a forgotten reference is worked
  out from the value under the key, never from a stored copy). A rule written before sealing existed, one written
  since, one written before a release and one written before a forget all still erase their message. When a form
  cannot be worked out because the key is unavailable, a message of a session with such a rule is not admitted:
  turn capture answers 503 before anything is reserved and the client sends it again later. An erasure by exact
  text finds the sealed copy without sealing anything new.
- **The gateway** (Hermes plugin, `pre_gateway_dispatch`, after the `/mind` gate): a message in which the plugin's
  copy of the detector finds a value is sealed through `POST /v1/secrets/seal` and rewritten before Hermes sees
  it, so Hermes' transcript, the model provider and session search never receive the value. So are the words
  Hermes puts before the model with it that the sender did not write: the message it quotes (in its author's
  scope) and a channel's backfill (in nobody's). The copy is byte for byte the sidecar's detector (a test keeps
  them equal). When the sidecar cannot seal, each value is withheld:
  `[secret withheld: the store was unavailable; send it again]`; a value the owner released passes as it came
  (the seal answer marks its span). A message with no detected value never reaches the sidecar there.
- **Hermes tool output**: the plugin registers the vendor-token, JWT and webhook patterns with Hermes' redactor
  (additive to its own) and every value it fills for a use with Hermes' exact-value redaction, so a tool that
  echoes it does not put it back into model-facing output.
- **The plugin's turn outbox** keeps a delivered row's turn id and envelope digest and drops its payload; a row the
  sidecar rejected keeps its payload in that private file (the turn's only copy) for repair.

## Claims

A credential never becomes a claim. The claims validator refuses a statement whose quotation names one (a password,
a credential, a secret, an API key: `sensitive_evidence`), and one whose own subject or predicate does
(`credential_label`): every label the detector reads before a value (`password`, `wifi password`, `passphrase`,
`passcode`, `PIN`, the codes, phrases, keys and tokens above, `CVV`), `credentials` and `secret`, and every name the
detector seals a value under in an assignment or a JSON field (`API_KEY`, `AUTH_KEY`, a `TOKEN`, a `SECRET`,
`refresh_token`, `client_secret`), written with spaces, underscores or in camel case. A label is read word by word
(a secretary is no secret), and a name the detector reads as about a credential, never one, names none (its id,
address, kind, count or lifetime: `token count`, `AUTH_TOKEN_ID`, `password length`). Capture, the validator and the
display below share that one vocabulary. So "The wifi at the lake house is ..." never becomes a `wifi_password` claim
holding its value, though its quotation names no credential. The turn keeps its words, sealed as capture sealed them.

Claims formed before that rule can hold such a value. Wherever claims are shown (the owner console's memory pages,
`colonymind memory inventory` and `history`), each is read as its card shows it (`colonymind.secrets.claims`): its words
each alone, and its predicate (read by its words, `wifi_password` as `wifi password`) beside its value, bare and with
the value quoted, so a value of several words is read whole. A claim whose own subject or predicate names a
credential, as the validator reads it (`access_token`, `refresh_token`, `authKey`), has its whole value withheld
whatever it looks like, a weak password the detector would never seal included, and that value is withheld wherever
it stands as itself (never inside a longer word). A value either finds is withheld wherever the listing or history
holds it: the value, the quotation, a correction's words and every other claim shown with it.

## References and use

`[secret S-XXXXXXXX: <label>]`: eight base32 characters of an HMAC of the scope and the value under the store's
key, and a label from a fixed vocabulary (the vendor for a prefix token, the name of an assignment, the words of
a label, "card number", "URL password").

The store is `secrets-vault.db` in the state directory (0600): per reference its scope (contact and session),
rule, label, source turn, the value under AES-256-GCM (a fresh nonce per row, the reference and the scope as
associated data), creation, last use, uses, and whether it was released or forgotten. The key is 32 random bytes in
`secrets-vault.key` (0600, the state directory) or in the OS keyring. A backup copies the databases, so it carries
ciphertext and never the key unless `secrets.vault.key_in_backups` is on, and then only in an encrypted archive; a
restore without the key keeps every reference and reports its value unavailable.

**Scope.** A value belongs to the person whose message carried it. The owner resolves, uses and releases their
own; a contact's value is never resolved by anyone, and the owner may list its label and forget it.

**Use**, through the owner-only tool `colonymind_secret` (the owner's own direct session; refused for guests, cron
runs and workers):

| Operation | Effect |
|---|---|
| `list` | references: label, whose, state, dates, uses |
| `use <ref> to_env <NAME>` | stored as `NAME` in Hermes' own credentials (`save_env_value`) |
| `use <ref> to_file <path>` | the reference in that file is replaced by the value |
| `use <ref> to_login <origin>` | a Hermes vault login item for browser autofill (with the identifier) |
| `reveal <ref>` | the value sent to the owner's own chat by the plugin (the model never has it); in a session with no chat (a local one), put in a private file (0600) the owner opens, removed within half an hour; the model is told where, never what |
| `release <ref>` | a false positive: the value goes back where the reference stood, everywhere, and is not sealed again |
| `forget <ref>` | the value is deleted; the reference reads "a value the owner deleted" |

**Calls in the owner's session.** A `tool_request` middleware uses a value in a call when its reference was
sealed from the owner's own words in their latest message in that same chat (the gateway hook saw it sealed from their
text; a quoted message, a channel's backfill, an expanded file, a pasted reference or their words in another chat
never count): their words stand behind the call. Any other reference becomes a one-tap ask bound to the digest of
the complete call (its tool and every argument, never a prefix), with where the value goes shown in full and the
whole call kept with the ask for review; the plugin puts it to the owner in their chat directly, the call waits
(`pre_tool_call`), and after their yes that same call is served once. A different call, however similar, is asked
about anew.

Where a value goes. A value is text only in a send's or a write's words (a message a person reads, a file's
content) and in a waiting process's input, as the whole of it (answering a prompt). **A value is never put into a
command's or code's text.** A terminal command or Python code names a variable instead: `$COLONYMIND_SECRET_XXXXXXXX`
holds the value of `S-XXXXXXXX` and `$COLONYMIND_SECRET_XXXXXXXX_FILE` names a private file holding it
(`os.environ[...]` in Python). The plugin sets both for that one call and nowhere else: the command runs in a
subshell that reads the value from its file into the variable, and the code runs with the value in the
interpreter's environment until it ends; the command or code itself is never read or changed (a command runs as it
was written, after the lines that set its values; code as one string literal), so no quoting, comment or construct
in it matters and quotes, `;`, `$(...)` or backquotes in a password are bytes, never syntax. A command that closes
the subshell early runs the rest without the value. Nothing outlives the call: not the session shell, the environment
Hermes keeps between commands, or the session's interpreter (a `cd` or `export` in such a command stays in its
subshell too). A reference written into a command or code is refused, and the call is told which variable to name.
What a program does with the variable is its own: a command that writes `$COLONYMIND_SECRET_...` inside another
program's source text has its shell put the value there, as it would any variable, which is why the ask shows the
command in full and the tool tells the model to keep it a variable. The files are 0600 in a 0700 directory; the
program that takes one records its process and removes it when it ends; the plugin removes one when its call is
over and no program holds it, when the program that took it is gone, or when none took it within an hour; never
while the program that took it runs, however long. A value that cannot be read stops the command before it runs.
Values reach commands and code on the local terminal backend only: elsewhere the value would have to travel as
command text, so the call waits and says to store it with `use to_env` and forward it with the backend's
environment passthrough. A `write_file` or `patch` in their own session is local and served without an ask; one
whose reference sits where no value goes (a patch's `old_string`, a path) waits.

The local terminal is the owner's too: `colonymind secrets list|show|use <ref> to_file|to_env|release|forget|
migrate`, which never print a value.

## What leaves the outbox

`mind/audit.py::outgoing_text` decides the words of every message the mind sends:

- to the owner: the stored words as they are; the body resolves an owner reference just before it claims the send,
  and a message whose reference cannot be resolved for now waits for a later tick (nothing leaves without it);
- the owner's dictated words to someone else (a `commitment_notice` formed only from their confirmed request): as they
  gave them; the body resolves a reference only when it is in those words and was sealed from their own message;
- anything else to anyone but the owner (a check-in, an outreach, a composed follow-up, a worker's message turned
  into an intention): every detected value and every reference is replaced by "(a private value, withheld)".

Without the owner's id the recipient counts as not the owner, so a caller that forgets it can only hide. What the
owner is asked about and what leaves are one string (`frozen_for_ask` uses the same predicate). A worker's own
send in a mind run is hidden the same way by the plugin's middleware, so the direct path cannot bypass it.

## Configuration

`secrets:` in `colonymind.yaml`:

| Key | Default | |
|---|---|---|
| `secrets.capture.enabled` | `true` | seal wherever a turn is stored |
| `secrets.capture.ingress` | `true` | seal gateway messages before Hermes sees them |
| `secrets.capture.rules` | every rule | a list of the rules to run, or a mapping `rule: off` |
| `secrets.vault.key_store` | `file` | `file` (`secrets-vault.key`) or `keyring` |
| `secrets.vault.key_in_backups` | `false` | an encrypted backup also carries the key |
| `secrets.migrate.on_start` | `true` | the first start seals what was stored before (once) |

A malformed section never weakens capture: the defaults stand and the problem is logged, in the sidecar and in the
Hermes plugin alike (both validate it the same way, a test keeps them in step), so a misspelled rule list never
leaves the gateway with no rule.

## Migration of what was stored before

`colonymind secrets migrate` (`--dry-run` prints counts per store, rule and value shape and writes nothing); with
`secrets.migrate.on_start` the first start runs it once in the background, guarded by a marker in
`secrets-vault.db`. Idempotent and resumable, one turn per transaction. It seals strictly: when the vault cannot
take a value (a locked keyring, an unreadable key) it stops without changing that turn or row, writes no marker,
and resumes at the next run; nothing is ever withheld over a stored value. A JSON value is decoded and its strings
sealed, each read with its field name too, so a value is stored as its bytes (a key block's line breaks, not their
escapes) and a value named only by its field is found. `release` puts the value back in every field sealing reached.

1. `turn_sources`: each message is sealed, the turn's digest recomputed, its lexical index rewritten and its vector
   job queued again. Each changed identity (a message hash, the turn's version, its content digest) is recorded as
   an old and new pair. Historical tool-observation rows receive the same source text sealing;
   their source IDs are retained, without observation-specific digest repair.
2. Lineage: every record in the memory stores that cites an old identity (annotations, claims, facts, a child
   turn's source references) is re-pointed to the new one, to a fixpoint.
3. Every other text column of the memory stores is sealed in place (queue rows, ingestion responses, the
   communications log, the mind's rows), in the scope of the row's person, or the owner's for the mind's own
   stores. A store that is not memory (approvals, audits, pairing) is never touched; an append-only table keeps
   what it holds and is reported. Then the ledger's derived indexes index what the pass queued: each claim a
   changed message states is located again in the claim key index. `release` ends the same way.
   Owner-goal tables are excluded from these generic walkers. An explicit owner-store sealing pass encrypts
   their content fields with the same vault while preserving original audit and receipt hash preimages.
   The repair requires a new completion marker, verifies all owner content and retires old WAL pages before
   marking completion. See [owner-goal storage integrity](OWNER-GOALS.md#storage-sealing-and-integrity).
4. `secrets-migration-report.json` (0600, counts only): what was sealed per store and rule, the nightly backups
   that predate the run (they roll off; removing them sooner is the owner's call), the legacy backend note, and a
   read-only count of Hermes transcript messages that still hold a value. Hermes offers no in-place scrub;
   `hermes sessions delete <id>` removes a session, and that is the owner's call.

A replay of a turn in flight at the upgrade answers as a conflict and the plugin marks it undeliverable: the turn
is already recorded, nothing is lost. Rolling back: the previous release reads references as plain text, and
`release` restores any value.

## CI

`scripts/secret_scan.py` runs the detector's form rules (tokens, key blocks, JWTs, webhook and credential URLs, bot
tokens, card numbers) over every tracked file and fails on a finding, printing its place and rule, never the
value. A synthetic fixture is allowed only where it is marked: a file whose first ten lines carry
`secret-scan: synthetic fixtures`, or a line carrying `secret-scan: synthetic`. A real value is never marked; it
is removed and rotated.

## Limits

- A secret typed in the CLI, spoken in a voice note (transcribed after the gateway), printed by a tool, read from a
  file an `@` reference expands, or sent while Hermes is busy with the session (a queued or steered follow-up
  never reaches `pre_gateway_dispatch`) reaches Hermes' transcript; ColonyMind seals its own copy, and Hermes'
  redactor catches the vendor-token families.
- Nightly backups taken before the migration hold values until they roll off.
- The mind cannot use a sealed value in its own tasks (the tool is the owner's).
- A command or code uses a value through its environment; what it then does with the variable is its own (a command
  that writes the variable into another program's source text splices the value there). The ask shows the command
  in full; a value written into a file with `write_file`, `patch` or `use to_file` is that file's content, and a
  file that is later run as a program carries it as its text.
- Values reach commands and code on the local terminal backend only.
