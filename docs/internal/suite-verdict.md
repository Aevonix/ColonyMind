# Local suite verdicts

`scripts/suite_env.sh run` uses an existing Python 3.12 environment. Set `PYTHON312`, `SUITE_VENV`,
`HERMES_SRC`, `HERMES_PYTHON` and `SUITE_PRIVATE_ROOT` as described in the script. The last directory and
its non-system ancestors must be private to the current user. No dependency installation is needed for
`run`. The supplied Hermes interpreter must have versioned distribution metadata and must not be marked
as a template. Each run copies that installation before tests can modify it.

The runner builds a temporary package overlay from the checkout's packaging map and prints the resolved
`__file__` paths before counting tests. An environment with a `SUITE-TREE` record for a different checkout
is refused. An unmarked dependency environment can be reused without changing its editable installations.
External model caches and pip run offline; live-sidecar variables are cleared. Tests needing git history
use the checkout's recorded objects, or skip with the missing commit or object in their reason.
An existing `PYTHONPATH` follows the checkout and Hermes paths, so a caller can expose libraries from
another supplied environment without installing anything. The Hermes source must match the installed
distribution's recorded editable URL or wheel hashes; a source-only path does not establish provenance.

If the sandbox blocks asyncio's socketpair wakeup writes, set `COLONYMIND_TEST_POLL_LOOP=1`.
Both phases then use the trunk's 1 ms selector polling mode. This only wakes test event loops;
it does not permit network sockets or replace failed integration tests with passes.

Both suites must finish and write complete JUnit reports. An interrupted phase returns 2 and has no
valid verdict. Completed runs compare failing test IDs with `suite-known-failures.json`; its `failures`
object maps every accepted ID to a one-line cause. New failures return 1. So can an abort before the
tests run, such as an unset variable; the verdict line, which only a completed run prints, tells the two
apart. A completed run ends with:

```text
== known failures: N of M still fail; X new; Y now pass
```

A run that the caller narrows (`-k`, `--deselect`) or stops early with an option (`-x`, `--maxfail`) is
judged only on what ran: "now pass" means "did not fail in this run", whether or not the test ran.

The failing test IDs come from pytest's own test reports, which `scripts/suite_results.py` records in
both phases, never from terminal text, so output options do not change them. An ID is the complete node
ID, parameters included; a recorded ID covers only the identical ID, never a longer one that starts with
it. A test whose setup, call, teardown or subtest failed is one ID; skips and expected failures are not
failures.

A run returns 2 without counts when a phase ran no test, when a call of `pytest.exit` ended a phase's
session before its normal end (whatever status the call named), when a JUnit report's declared failures,
errors or skips are not those of its test entries, or it declares tests without holding an entry or the
reverse, or fewer tests than its entries name (as a run under `--setup-plan` or `--setup-only` does, in
which no test body ran), when a failure or error in a report is not a recorded ID or a recorded ID is
not a failure or error there, when a module could not be collected (whatever status pytest gave), and
when the record is not one JSON object whose `failures` member maps IDs to text causes, or holds a key
twice. One line on standard error says why. The line "the session was ended early by pytest.exit" names
that call and not its caller, which can be a test, a fixture, a hook, or pytest itself when its debugger
is quit. The declared test count may be above the number of tests the entries name, as pytest's own is
for subtests, for an expected failure whose teardown raises and, in older versions, for a teardown error
after a reported call.

The output directory holds both phase logs, JUnit reports, recorded results and the failing ID list.
Now-passing IDs are reported too; remove them from the record only after inspecting a complete run. Do
not add failures introduced by a change to make its verdict green.

## Optional daily run

`scripts/daily_suite_verdict.sh CHECKOUT VERDICT_LOG` runs the same runner and appends one UTC date,
commit ID, verdict and counts to the log. `INCOMPLETE` records cannot be mistaken for passing tests.
There is no notification, network call or dependency installation in the daily wrapper.

Every start of the wrapper with its two arguments leaves one line, unless the wrapper is ended before it
has written the line: by a signal other than SIGTERM, by a SIGTERM that arrives before the wrapper has set
its handler, or by an error on which bash itself ends it, such as a `tail` that cannot read the run's
output. A start that is ended in one of these ways leaves no line. A start that cannot run
(`SUITE_PRIVATE_ROOT` unset or not a directory, a checkout path that cannot be entered or in which git
finds no commit, a failing `mktemp`) and a run that ends without a verdict (the runner ends without one
final verdict line or with a status that does not fit it, or the run is stopped with SIGTERM) write
`verdict=INCOMPLETE` followed by a quoted `reason`. The exit statuses are the earlier ones: 2 for a run
without a verdict, the failing step's own status for a start that cannot run, and the signal for a stopped
run. The line goes to the verdict log and to standard output; when the log cannot be written it goes to
standard error instead, and a run that reached its end has status 1, as before.

The full output is kept under `SUITE_PRIVATE_ROOT`: the runner's directory `colonymind-suite.XXXXXX` and
the wrapper's file `daily-suite.XXXXXX.log`. At the end of a run the wrapper keeps the three newest of
each, by modification time, and removes the older ones that were there when the run started. It removes
nothing else: only an entry directly under that directory, a directory with the first name or a regular
file with the second, never a symbolic link and never the output of the run in progress. The three that
stay are chosen by modification time alone, so output dated ahead of the clock is kept in place of newer
runs until the clock passes it. Keep the verdict
log outside `SUITE_PRIVATE_ROOT`; the wrapper never removes the file that is its verdict log, whatever its
name. Runs of the runner made by hand under the same directory are counted and removed like the daily
ones, so the daily run needs a directory of its own. A removal that fails is named in that run's line as
`not_removed=NAME` and does not change the verdict or the status.

The example user units in `scripts/systemd/` run once a day at 04:00 local time, with one catch-up run
after downtime. The service manager stops a run that takes longer than three hours; the wrapper then
leaves its `INCOMPLETE` line if it can still write one. The units are not installed or enabled by the
scripts. To use them, put the service and timer
in the user's systemd unit directory and write `~/.config/colonymind/suite-verdict.conf` with absolute values
for the five environment variables above plus `SUITE_CHECKOUT` and `SUITE_VERDICT_LOG`. Create the private
output and log directories first. Then reload the user manager and enable `colonymind-suite-verdict.timer`.
Inspect `systemctl --user list-timers` and the dated log to confirm execution. Disabling the timer stops
future runs; existing logs and artifacts remain available.
