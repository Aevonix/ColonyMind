# Session standing and unknown authority

Before this change, unknown standing escaped the guest branch into the
permissive return. The guard now resolves standing before selecting any tool
lane. Unknown or failed resolution produces the guest-style final refusal and
a WARNING containing `standing_unknown`, the tool name and a fixed session
class. No ids, arguments or exception text are logged by that refusal.

Unknown sessions receive no tools. A session without a verified sender or
registered parent has no established conversation boundary for files, memory
or messaging. This covers the whole tool namespace, including both MCP naming
conventions, browser vault tools, future plugins and guest opt-ins. Established
guests retain their existing tool policy. Independent verification of the
current direct sender can still establish owner standing if another resolver
fails; a stale group speaker or unresolved parent cannot.

## Creation and inheritance

| Path | Standing source and retained record |
| --- | --- |
| `plugins/hermes-plugin/capture.py:305` (`SessionMap.observe`), `:294` (`inherit`) | Existing explicit snapshot; otherwise the creating parent. Observed terminal/system roots record their trusted host standing. Other senderless roots require the automation resolver. |
| Pinned Hermes `tools/delegate_tool.py:274`, `:330` → `plugins/hermes-plugin/guard.py:627`; hook registration in `__init__.py:179` | Snapshot of the initiating parent turn before child execution, persisted by `SessionMap.inherit`. Both ordinary and nested delegation use this hook. |
| `plugins/colonymind-hands/dispatch.py:285`, `:558`, `:598` | The verified admission gate records the parent's actual `owner`, `mind` or `system` standing in the hand plan and durable spool metadata before dispatch. Guests and unresolved parents cannot enter this creation path. |
| `plugins/colonymind-hands/dispatch.py:771`, `authority.py:468` → `plugins/hermes-plugin/guard.py:552` (`register_hand_sessions`) | Bind the hand's saved standing to each child and rotation. Registry membership alone is insufficient. Restoration at `dispatch.py:239` reads saved metadata; an older spool requires a resolvable parent. |
| Pinned Hermes `cron/scheduler.py:2553`, `:2468`; `plugins/hermes-plugin/guard.py` (`_remember_creator`, `_report`, `_cron_origin`); `capture.py` (`observe`) | Successful cron creation/update persists the initiating caller's explicit standing beside its parent. A run records the registered job origin when observed. The legacy sidecar route proves mind provenance only: its default owner classification grants nothing. The cron context marker alone grants nothing. Missing jobs and unavailable ownership remain unknown. |
| `plugins/hermes-plugin/body.py` (`_create_task`) → `guard.py` (`automation_standing`, `mind_run`) | A new mind dispatch persists `mind` before its board transaction commits. A profile or idempotency key alone grants nothing, including on retries of older tasks. Saved owner, guest and unknown tasks keep their creation standing on every profile and on the exception read path. |
| `plugins/hermes-plugin/guard.py:963`, `:586` | Successful ordinary kanban creation records the parent's standing. A later worker reads that exact task record; a task id or ordinary profile alone grants nothing. |
| Legacy `colonymind_task` | Its platform/controller creation code was removed before this base. There is no active creation path to patch or silently re-enable. Historical replay requires retained host admission evidence for the exact native session and configured owner contact; the platform name alone grants nothing. |

The host-private SQLite standing store contains session/job snapshots and their
parent reference. It belongs to the configured instance, so profiles share its
authority and separate instances cannot inherit each other's records. Guest
and mind reads or writes of this authority file trip the same protection as
the instance key, including reads through aliases. It is separate from model-supplied arguments and survives
process restarts. Compression resolves the nearest saved session snapshot,
including explicit unknown, through Hermes' durable ancestry before recording
the hook's parent reference. That reference may still name the original group.
The recovered snapshot is persisted under the new identity before any tool call.
Existing session snapshots cannot be overwritten by repeated inheritance.
A child without a creation hook needs an immutable ancestor snapshot; a mutable
group's later sender cannot supply its missing creation evidence. Parentless,
unregistered sessions remain unknown.

## Groups

Owner standing belongs only to the current sender's turn. Replacing a group's
sender replaces the sender's `SessionInfo`; binding caches also include sender,
platform and chat. A currently bound guest cannot use an earlier owner's
cached standing, including during resolver failures. A child snapshots its
creating turn, so later owner/guest interleaving cannot upgrade that child.
Cron and board actions are normalized before recording. Their initiating tool
call's standing is stored durably before execution and consumed on completion,
even after restart or more than 2,048 other outstanding calls. Failed results
also consume their record. A missing call identity or missing initiating record
persists unknown; completion never substitutes the group's current standing.
Post-tool provenance reports can still use a positively recorded mind worker's
own task, but those reports do not replace missing initiating authority: the
created job or task keeps its unknown standing.
Delegation retains the initiating parent turn durably for all of its children
rather than following a later speaker, even after restart.

Owner continuity is recovered from positive creation evidence: saved session
ancestors, saved task/job creators and outstanding initiating turns. Newly
authenticated direct owner turns can establish their own standing. Recovery
does not infer that a parentless API root or an unretained cron job was owner
work. Historical calls without those records need independently authenticated
admission evidence for that exact creation; their authority cannot be recreated
from a current group speaker, a profile, a context marker or a job's absence.
Such refusals remain an owner liveness gap when historical owner work cannot be
proven, rather than evidence that the refusal was harmless.

For forward recovery, a currently authenticated owner can explicitly update a
retained cron job through the guarded native tool, or recreate an unretained job.
The successful operation publishes that new admission's owner snapshot. A new
delegate likewise takes the authenticated owner's initiating turn. This restores
future work without certifying missing historical admissions or upgrading an
existing explicit unknown session. Do not backfill old owner records from the
legacy cron route's default classification.

## Read-only evidence and replay

`scripts/standing_fixture.py` reads the retained hand registry and tombstones,
task rows, kanban, initiatives, cron ownership and the last 30 days of Hermes
assistant tool-call names. It also reads exact-session native owner admission
records for retired task sessions. SQLite access uses `mode=ro`; no host
application code is imported and no model is called. The optional
`--read-cron-standing` uses only the existing read-only cron-origin GET route
after the operator verifies its availability. Credentials stay in memory.

Identifier joins remain on the host; salted hashes remain in memory and only
new synthetic labels leave it. No messages, arguments, identities or hashes
are exported. `scripts/replay_standing.py` feeds these explicit snapshots and
tool names through the patched guard without network or tool execution. This
checks the standing boundary, not effect/recipient/budget verdicts that require
arguments. Mind calls, if present, are reported as requiring that separate
check rather than being silently treated as owner calls.

Keep deployment measurements in the private replay artifact. Report total calls
by class, established owner calls and their refusal count, and every residual
call by synthetic ordinal, class, tool and missing-parent reason. A parentless
API root without admission evidence or a cron job no longer retained cannot
be certified as legitimate owner work. Unresolved sessions must remain denied;
never silently reclassify them to reach the pass threshold. Replay does not
change or backfill live state.

## Unreadable ancestry and process evidence

An unavailable parent does not establish a cron context. Outside a positively
identified cron run, resolution retains the session's own standing. Within a
cron context, an unreadable or unregistered origin remains unknown and gets
no tools. The bound-standing cache continues to include the current platform,
sender and chat; detached work continues to use its creation-time snapshot.

Registry failures emit `ancestry_unreadable` warnings with only the exception
class and increment the process counter of that name. Adapter `/mind status`
exposes the counter. Standing fallbacks and unresolved senders warn; automation
authority decisions log their tool, fixed source, origin and reason at INFO.
These diagnostics contain no session/sender identifiers, text or arguments.

At registration the adapter atomically replaces
`<instance>/hermes-plugin-registration.json` with a single-link 0600 record.
Schema 1 contains exactly `pid`, `process_start` (epoch seconds), `module_path`
(the resolved guard module), `interpreter`, `hooks`, `written_at` and `schema`.
The hook list comes from successful registrations, including helper hooks.
The checker binds this evidence to the running process and start time; there
is no heartbeat. Failure to publish evidence warns and leaves the checker
unverified. No identity, configuration content or credential enters the record.

## Other pre-tool vetoes

The coexistence tests register a Companion verdict fixture alongside the actual
adapter hooks and call Hermes' real pre-tool dispatcher in both orders. Either
plugin's block wins over allow or approval. This verifies composition only;
the fixture does not implement Keychain detection or HOME notices.

The instance-key tripwire (`reaches_key`) covers reads/searches that reach the
configured ColonyMind key, plus the standing store via its caller. Companion's
Keychain rule covers a different credential store and also blocks owner calls;
the ColonyMind key rule preserves the owner's own lane. Generic credential-read
tool families can intersect, so a single tool call may receive both vetoes.
Keep credential-specific detection and notices with their owning hook; the
dispatcher returns one blocking result and never turns another hook's allow
into an override. No Keychain rule or duplicate notice sender is added here.
