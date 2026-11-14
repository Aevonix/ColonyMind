# Companion device capabilities

This page states the backend contract for Companion's device capabilities.
Companion owns native UI, OS consent and execution.
The initial capability groups are `health`, `alarms` and `home`. There is no new
service, scheduler, feature flag, pairing authority or native adapter. The
owner's location report, `POST /v1/host/owner/location`, is a separate contract;
these routes do not track location.

Reuse the UUID `device_id` from Companion's existing authenticated registration.
The hub derives this identity from its session principal for reports, claims and
results; it must not trust a phone-supplied identity. Owner device management can
explicitly address another registered device. Companion retains its existing
session revocation, rename and pairing flows. On revoke/disconnect, the hub must
report `state: disconnected` here before considering backend cleanup complete.
An online state is the app's last report, with `reported_at`, not a server presence
guarantee. Repair reports use the same ID. Omitted capabilities lose their grants;
if advertised again they default to `ask`.

## Reports and grants

App routes require an authenticated, issued owner relay credential with explicit
`owner: [companion]` and `manage` access. An all-platform grant, guest credential,
the adapter's instance key and the separate `answer` credential do not qualify.
Known Hands sessions cannot mutate these routes. This follows the explicit
Companion owner-platform check used by the owner-location route, and also
excludes the adapter key so a model cannot report OS grants or authorize itself.
The hub keeps these credentials server-side; none belong in phone payloads.

All paths below are relative to `/v1/host/owner/devices`:

| Method and path | Contract |
| --- | --- |
| `GET /` | `{devices: [...]}`; inventory and each permission summary |
| `GET /{device_id}` | Full report, capabilities, authorizations, revision and report time |
| `PUT /{device_id}` | Replace the app's report; preserve existing agent authorization |
| `GET /{device_id}/capabilities/{capability}` | One capability and the device revision |
| `PATCH /{device_id}/capabilities/{capability}/authorization` | `{expected_revision, method: read\|write, mode: allow\|ask\|deny}`; updated device |
| `GET /{device_id}/invocations` | `{invocations: [...]}`; reconcile pending and historical requests |
| `POST /{device_id}/invocations/{id}/claim` | One dispatch claim; returns the exact invocation; a second claim is `409` |
| `POST /{device_id}/invocations/{id}/result` | `{ok, payload?, error?, card?}`; updated invocation |

`PUT` takes only the following shape (the ID is in the path):

```json
{
  "display_name": "Study iPhone",
  "platform": "ios",
  "app_version": "1.0",
  "state": "online",
  "capabilities": [{
    "capability": "alarms",
    "os_grants": {"read": "granted", "write": "not_determined"},
    "commands": ["alarm.list", "alarm.set"]
  }]
}
```

State is `online|offline|error|disconnected`. An OS grant is
`unknown|not_determined|granted|limited|denied|restricted|unsupported`; an omitted
method is unknown. Only `granted` passes the backend OS check. Report only the
commands this build and OS can actually execute. Native handlers still validate
parameters and recheck the exact OS scope immediately before execution, including
HealthKit's limits on observing read authorization. Do not infer grants from a
successful system prompt. Report truthful unknown/unavailable states instead.

Every capability has independent `authorization: {read: ask, write: ask}` by
default. `deny` refuses; `ask` requires a fresh exact-call approval; `allow` is the
owner's standing authorization for this device, capability and method. All three
remain independent of OS permission. Never/When using/Always or On/Off labels for
OS settings must not be mapped onto agent modes. OS reports cannot set authorization, and
grant edits cannot change OS state. Edits require the latest revision (`409` on a
stale edit). Report changes also advance that revision; identical reports only
refresh the report timestamp. Changed device revisions invalidate unclaimed
invocations, including an already approved one. Submit a new reviewed ID to try
again after a changed report; never silently retry an uncertain dispatch.

Command vocabulary is intentionally small. `health.read`, `alarm.list` and
`home.read` are reads; `alarm.set`,
`alarm.update`, `alarm.dismiss`, `alarm.dismiss_all`, `home.execute` and
`health.write` are writes. Each must be explicitly advertised. Parameters are
bounded opaque JSON owned and validated by Companion's native adapter; command
names do not promise native implementation or Android background behavior.
Health ingestion/backfill and background alarm delivery are outside this change.

## Model requests, approvals and results

The adapter tool `colonymind_device_invoke` offers only `inventory`, `request` and
`status`. Its backend routes require owner authority and the existing `manage`
operation. A guest cannot reach them. There is no model operation for reports,
grants, approval answers, claims or results:

- `GET /v1/mind/device-capabilities` returns the registered device inventory.
- `POST /v1/mind/device-invocations` accepts
  `{id, device_id, capability, command, parameters?, deadline}`.
- `GET /v1/mind/device-invocations/{id}` reads/reconciles one record.

IDs are stable, at most 96 ASCII letters/digits/dot/underscore/hyphen, beginning
with a letter or digit (Companion UUIDs work). The caller chooses a unique ID and
retains it across an uncertain HTTP response. Reusing it with changed parameters,
device, capability, command or deadline returns `409 invocation_id_conflict`.
The timezone-aware ISO deadline must be future and at most one hour away.

Each persisted record includes `id`, `device_id`, `capability`, `command`,
`parameters`, `deadline`, `device_revision`, `mode`, `approval` (an independent ask
ID or null), `state`, `result` and timestamps. Records use the existing Mind state
store. OS permission and command availability are checked before forming any
approval. Existing authority off/deny rules still apply to standing allows.

In `ask` mode, the backend constructs an exact effect covering the invocation ID,
device, capability, command, full parameters, deadline and report/grant revision.
The effect identity hashes all these terms. Its recorded call arguments contain
only the ID, device, capability, command and a SHA-256 parameter digest; parameter
keys and values stay in the invocation record used for native dispatch and never
enter approval, audit or autobiography records.
It reuses `Mind._hold`'s non-executing hand-origin `floor_effect` note, the existing
owner-answer reader, `Authority.effect_verdict`/`permit_for`, and the existing
`permit_used` recorder. No Hands executor or duplicate dispatcher is created. A
generic task approval and a claimed or operator yes cannot authorize the device.
The existing Hands permit controller and its exact-call checks stay in place.

Companion reads the ask through `GET /v1/mind/permit-asks/{approval}` and answers
through `POST /v1/mind/permit-asks/{approval}/answer` using its existing separate
`answer` credential, `Idempotency-Key` and `{choice: yes|no, platform: companion,
contact_id: <owner>}`. Existing owner-message answers work too. The model's
credential cannot answer. Approval never changes OS grants or standing device
authorization. This backend does not reuse another invocation's recent decision.

The state progression is `requested → awaiting_approval → ready → dispatched →
succeeded|failed`. `allow` skips `awaiting_approval`. Other states are
`os_permission_required`, `unavailable`, `denied`, `cancelled`, `timed_out`.
Reads reconcile approval, revision and deadline and close invalidated asks; no background timer is needed.
Before claim, the backend rechecks availability, OS report, revision, deadline,
off/deny policy and approval undo. At claim it validates the exact, unused permit
against `mind.floor.permit_hours` and consumes it through `permit_used`. Reading
an invocation as `ready` consumes nothing and cannot extend the permit's expiry.
Only a successful claim authorizes the phone
to execute, once. The app obeys the same deadline and rechecks OS permission locally after claim; an OS
failure is returned as a result with `error: os_permission_required` and an
optional permission/settings card, not as agent approval.

Failure requires an error code; success forbids one. A result can be supplied only
after dispatch. Identical results replay safely; conflicting ones return `409`.
A late result is retained with `late: true` and `state: timed_out`. Expired,
already claimed or ambiguous requests are never dispatched again. After an
uncertain claim response, read the record and reconcile locally; a `dispatched`
record alone is not permission to execute again. An expired dispatched command
may already have run. Revocation cannot retract native work already claimed.

Reports and payloads reject extra fields, recognized credential/phone content,
credential-bearing keys and bodies over 64 KiB. Validation responses omit input
values. Send no secrets, credentials, phone numbers or signed credential URLs in
metadata, parameters, errors, cards or results. Native adapters must return only
the result data needed by the owner. No device payloads are logged by this code.
