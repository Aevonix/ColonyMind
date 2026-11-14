# Job watch

`colonymind --instance /path/to/instance watch --once` runs one short-lived
process. Schedule it independently every five minutes using the examples in
`scripts/launchd/` or `scripts/systemd/`; replace their absolute paths before
installing them yourself. No timer is installed by this feature. There is no
sidecar worker or dependency on the sidecar service. If the sidecar is frozen or
down, watch continues reading local records and explicitly reports that it is
still running.

Watch reports and never stops, restarts, signals, adopts, or deletes jobs. Every
source SQLite database uses `mode=ro`; native scheduler and store initializers
are never called. Only `<instance>/jobwatch.db` is writable, created with mode
0600. Its findings, snapshots and notice queue contain identifiers, hashes,
counts, times and subject names, never full commands, arguments, prompts or
output. Completion output is transiently parsed to
hash its last non-empty line; live checkpoint output, when present, is hashed
as a whole buffer. Neither is logged or stored.

```yaml
watch:
  enabled: true
  hermes_home: ~/.hermes
  stuck_minutes: 30
  repeat_failures: 3
  repeat_window_hours: 24
  health_timeout_s: 10
  dependency_failure_passes: 2  # Consecutive failures to open; first success closes.
  notice_timeout_s: 30  # Kill the sender process group; retry timeouts after one hour.
  notice_command: []
  downloads:
    url: ""
  endpoints:
    - name: local-service
      url: http://127.0.0.1:8080/health
      expect_status: 200
  tmux: true
```

The selected Hermes home supplies `processes.json`, retained
`logs/process-results/proc_*.json`, `cron/executions.db`, `cron/jobs.json`,
`kanban.db`, and `kanban/boards/*/kanban.db`. The instance supplies `hands.db`.
Set `watch.hermes_home` explicitly for another Hermes profile. No other profile
or arbitrary directory is scanned. Completion receipts are retained by Hermes
for seven days or 64 files; watch does not change retention. Their modification
time is the completion time because the current receipt format has no end time.

The owner-only turn-start work list and `GET /v1/host/executions` read the same
Hermes process records through the same reader and `watch.hermes_home` setting.
Their `native_processes` view includes running processes and failures that ended
within the last hour, with program names only, state and timing. Arguments and
output are excluded. This reader opens only process checkpoints and completion
receipts; it does not scan cron or kanban. A missing, unreadable or timed-out source
gives an empty list with unavailable coverage; the turn-start reader has a 150 ms budget.

Checks:

- No CPU or output change for the configured idle period. `ps` measures a
  process tree, including detached tmux panes. Linux `/proc` write counters
  measure output conservatively (all writes count as activity). Tmux provides
  history size/bytes and cursor metadata. Detached panes whose foreground command
  is `sh`, `bash`, `zsh`, `fish`, `dash`, `ksh`, `tcsh`, `csh`, or `login` are excluded
  as idle shells when `ps` reports that the terminal foreground process group is the
  shell's own, even with background jobs or helper children. Explicit pane commands,
  including `./job.sh` and `bash job.sh`, remain watched. If `tpgid` is unavailable,
  watch falls back to excluding only shells with no children. CPU activity prevents a busy compile
  from being marked stuck. Missing activity measurements mean unknown, not idle.
  On macOS and other systems without `/proc`, watch retains `ps` CPU time and
  pairs it with a digest of the process's checkpoint `output`, when present;
  one-shot cron's `run_claim.at` and `by` in `cron/jobs.json` (only with one active
  attempt for that schedule); or kanban's per-run `last_heartbeat_at`.
  Process-tree start times are included, so PID reuse restarts the idle interval.
  Hands inherit activity from their owned process when available at the initial
  process probe. A missing observation or long sampling gap restarts the interval.
- Consecutive failures of a redacted command signature with the same last-error
  hash. Success, a different error, or an unknown error breaks the streak.
  Repeat findings close when their failures age out of `repeat_window_hours`;
  the notice says there have been no new failures in that window. Only a later
  successful run of the same command is described as cleared. A changed error or
  a shortened streak is described as no longer meeting the threshold.
  A later qualifying streak opens a new finding.
- Declared runtime limits for kanban runs, cron declarations (when supplied),
  and hands; hand turn budgets when usage is recorded. No inferred default
  budget is applied. Current native cron execution rows and kanban run rows do
  not expose turn consumption, so they cannot prove a turn overrun.
- Optional qBittorrent-compatible `/api/v2/torrents/info`: unfinished downloads
  with no connected or known sources for an hour. Unset, unreachable or invalid
  download endpoints are skipped, with failures logged by type and counted.
  Sampling gaps restart this interval.
- `/v1/host/health` must fail for two consecutive passes before one down/frozen
  finding opens (`dependency_failure_passes`); the first success closes it.
  `/v1/mind/status` produces at most one counter summary per UTC day, for lost
  work only: `context_writes_dropped` and `mind_incomplete.<task>`. Loop stalls,
  read failures and other counters do not trigger this notice. The first
  successful read establishes a baseline; a decrease is treated as a process reset.
  Claim-settling timeouts count only unfinished claims queued since the night
  started. Older unfinished claims remain visible in the night's `claims_backlog`
  count and do not trigger an incomplete-night notice.
- Optional endpoints must return exactly their expected HTTP status; redirects
  are not followed. Each is checked every pass and uses the same consecutive
  failure threshold and first-success recovery as sidecar health.

On macOS, current Hermes background processes are never judged stuck: Hermes
keeps their output in memory and its `processes.json` checkpoint has no `output`
field. Linux can judge them using `/proc` CPU and write counters. Recurring
cron runs have no per-run progress heartbeat; their ledger changes only on state
transitions and their output files are written at completion. They also remain
unjudged without `/proc`, as do kanban runs without a recorded heartbeat and
overlapping cron attempts. CPU alone cannot distinguish a hang from a network
wait. Failure receipts and repeat checks still work on macOS. Declared runtime
limits are checked independently of activity, though current background-process
records supply no limits; cron, kanban and hands supply them where declared.
Cron claim heartbeats can continue while work is hung, and unrelated work in a
shared scheduler process can advance
its CPU counter; these signals are conservative. A non-shell interactive program
waiting for input can still look stuck in tmux; only the listed shells are excluded.

A finding queues a notice on opening when `notice_command` is configured. A
finding opened without a command gets one opening notice on a later configured
pass if it is still open. Findings that close before an opening was queued send
nothing. Recovery notices require a retained opening notice; an expired opening
is never re-queued. Daily counter summaries have no recovery notice. At delivery,
more than three pending ungrouped openings never offered to the sender form one
grouped notice; pending closes
are grouped separately by the same rule, including notices from earlier passes.
Each group includes counts by check and up to three subjects using the same safe
wording as individual notices. Groups and their ids are committed before sending
and retained on retries; later notices do not join an existing group. Each pass
offers never-offered notices before retries across different findings, while each
finding's notices stay in queue order and any notice behind an older pending
notice waits as a single without using allowance. Within each of these sets, groups
come before singles, oldest first within each kind. Every eligible group is
considered, with at most one group and three singles successfully delivered per
state per pass. Failed or skipped offers use no allowance. A rejected notice or
one waiting for its hourly retry holds only its own findings; unrelated groups
and singles can still be delivered. A sender timeout ends delivery for that pass;
the next pass starts with never-offered notices again.
Unknown/unreadable job sources do not close
findings.
Job notices name the program (never its command arguments), job kind and local
start time as HH:MM. Sources without a program name retain their generic job
label; absent start times are reported as unknown. Endpoint notices name the
configured endpoint; download notices use the name returned by the downloads
API. Recovery notices retain the opening subject, even after it disappears from
the inventory. Sidecar notices name ColonyMind; counter summaries name each
increased counter and how many events are new at the time the summary is queued.
`notice_command` is an argv list executed without a shell; a short plain message
goes to stdin, with a 30-second deadline (`notice_timeout_s`) and discarded command
output. On timeout the sender's whole process group is killed. Configure
an independent delivery command to receive alerts during sidecar outages. There
is currently no general deduplicated notice API. An unset command queues no
notices and logs one warning per pass. Findings remain available in the findings
API and CLI; notice text is never logged.
Activity probes and dependency checks run before the write transaction.
Finding transitions and pending notices commit together before delivery.
Delivery is at least once: each notice is marked sent only after success;
failures remain pending for the next pass, except timeouts, which are retried
after an hour and hourly if they keep timing out. Before delivery, pending
openings for findings that have closed are discarded along with their pending
closes. Pending notices older than 24 hours expire and increment
`writes_dropped.jobwatch_notice_expired`. Legacy notices without a queue timestamp
use the finding's opening or closing time. Delivery is
serialized across passes. A crash after delivery but before marking it sent can
repeat a notice. Delivery failures log only the error type (plus the SQLite
error name when applicable) and increment `delivery_failures` in `jobwatch.db`
and the process counter `writes_dropped.jobwatch_delivery`.
Every delivery command receives `COLONYMIND_NOTICE_ID` in its environment, never
in the notice text. This stable id identifies the queued finding event (or grouped
events) and stays the same on retries. A single notice's identity is persisted
with its first delivery claim, before the sender runs; it can never be regrouped.
Opening, recovery and a later
reopening have different ids. A sender that records the ids it has delivered can
drop repeats, including delivery that succeeded before the sender timed out.

Invalid `watch:` settings do not stop the sidecar: it warns with the field name
and uses safe defaults, preserving a valid selected Hermes home. Only
`colonymind watch` rejects the invalid section.

Inspect without sidecar access:

```sh
colonymind --instance /path/to/instance watch --list
colonymind --instance /path/to/instance watch --once --dry-run
colonymind --instance /path/to/instance watch --dry-run --since 2026-10-28T00:00:00Z
```

`--since` filters retained completion records (and repeat history) by completion
time; live-job and dependency checks still run. `--dry-run` copies existing
watch state into memory, writes nothing and sends nothing, printing current
findings as JSON. A replay cannot reconstruct vanished receipts or historical
CPU/output samples. Dry runs neither deliver nor mark pending notices sent.
`--list` reads only the watch store and prints an object containing `findings`
and the cumulative `delivery_failures` count (zero before the first run). The owner-only
`GET /v1/host/watch/findings` route reads the same store read-only in a worker
thread. Its findings list remains empty if the watcher has never run. The regular CLI uses
the existing local-client API key helper; it never prints the key or reads Hermes
secrets or `config.yaml`.
