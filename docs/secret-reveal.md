# Companion saved-password reveal

The app runs Face ID before each explicit owner reveal. Face ID is enforced by
Companion; no client-supplied biometric flag grants backend authority.

- Route: `POST /v1/secrets/saved-password/reveal`, private HTTP only, excluded
  from OpenAPI, every MCP/tool schema and the Hermes adapter.
- Auth: the existing issued Companion owner relay credential (`owner:
  [companion]`, `access: [answer]`, `contacts: own`), using `Authorization:
  Bearer <redacted>`. This is the same boundary as explicit permit answers.
  The instance key (including rotations), `manage`, legacy, non-owner,
  tool-origin and model credentials cannot call it. No new grant or flag.
- Request: exactly `{ "id": "vault_0123456789ab", "session_id":
  "owner-session" }`. One current saved-login handle and the authenticated
  owner app session bound by the relay, never tool/model arguments. Session:
  1–256 ASCII letters/digits or `_.:-`, starting with a letter/digit. If
  `X-ColonyMind-Session` is supplied it must match. Registered hand sessions,
  including retired aliases in body/header/query, are refused before lookup.
  Repeated session selectors in the query or headers and duplicate JSON keys
  are invalid. The body is capped at 2,048 bytes before JSON decoding, using
  both Content-Length (when present) and the actual streamed bytes.
- Response: `200 { "id": "vault_0123456789ab", "password": "<redacted>" }`.
  Only that login's password; no username, OTP seed, other fields or bulk/list
  operation. Every response, including refusals, has `Cache-Control: no-store`.
- Errors: `401` missing/invalid authentication; `403` wrong credential,
  development mode or hand session; `422 invalid_reveal_request` for malformed,
  oversized or extra fields; `404 secret_not_found` for unknown/non-login/no
  password handles; `503 secret_reveal_unavailable` for vault/bridge/audit
  failure (`hand_session_unavailable` if the hand registry fails).
  Auth precedes lookup, so refused callers receive the same error for existing
  and absent IDs. Errors never echo request bodies or upstream diagnostics.
- Audit: one committed `sealed_uses` row in ColonyMind's existing
  `secrets-vault.db` before each successful response: `ref` = saved-login handle,
  `at` = UTC ledger timestamp, `purpose` = `owner_reveal`, `detail` = JSON
  `{credential, session_sha256}` (issued credential ID and SHA-256 of session).
  No value, token, username, phone number or raw session is written. A retry
  creates a new audit row; there is no response replay/cache. Audit failure
  returns 503 without the password. A disconnected client can still have a row
  because the server committed its release before sending.

Saved logins live in Hermes' profile vault, not ColonyMind's conversation-secret
references. The backend uses existing `hermes.python` discovery and
`hermes.home` selection to call `VaultStore.get_meta/resolve_secret` against
`<hermes.home>/vault` in Hermes' interpreter. The password crosses an anonymous
pipe into the HTTP response; diagnostics are discarded. No credential copy,
new service, vault format or key handling is introduced. Missing Hermes or an
unreadable store fails closed. `vault-captures`/`vault_fill_confirm` are
schema-only approvals; they never hold saved passwords.

Companion must authenticate its owner app session, complete Face ID on each
tap, and forward through its private relay credential. Keep that credential
outside model/tool environments; never forward their requests, log the
response, persist it or include it in chat/history. Hide and discard the value
when the detail is hidden, dismissed, locked or backgrounded. No backend can
infer who initiated a request from a stolen relay credential or verify Face ID
from a Boolean in a request.
