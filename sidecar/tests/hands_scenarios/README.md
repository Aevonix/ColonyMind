# Hands scenario suite

OS1 implements the fourteen section 9.3 scenarios and a guard mutation for each.
The socket-free tests use the existing fake Hermes derived from the pinned
delegation source. Durable controller cases use temporary SQLite stores. The
catalog's assertions are shared with the rc-clone replay gate.

Run from the repository root with the sidecar's test dependencies installed:

```sh
PYTHONPATH=sidecar python -m pytest sidecar/tests/hands_scenarios -q -ra
```

Every scenario has `guard-on` and `guard-off` variants. The latter must encounter
the named safety assertion after disabling the production guard. Setup failures,
missing observations and a different assertion do not count as killed mutations.
Controller mutations alter an exact branch in memory and are restored by pytest;
no production source is edited.

| Scenario | Guard | Additional integrator requirement |
| --- | --- | --- |
| 1. One-slot contention | Dispatch admission | Real shared cron dispatch |
| 2. Cancel mid-tool | Cancellation propagation | Actual stream and engine running-count decrease |
| 3. Restart with open effect | Unreconciled-effect admission | Local sockets; production probe shown in digest |
| 4. Duplicate completion | Result idempotency | Local sockets; OC4 evidence and autobiography wiring |
| 5. Notice timeout | Durable delivery id | Platform sink with its own delivery ledger |
| 6. Revised goal | Intent-version comparison | Local sockets |
| 7. Repository writers | Exclusive resource locks | Local sockets |
| 8. Engine fallback | Actual-route audio stripping | Actual provider switch |
| 9. Sidecar outage | Spool acknowledgement | Sidecar shutdown and recovery |
| 10. Owner during review | Owner-first ingress redirect | OC3 hooks; measured no-review baseline and owner turn-start timing |
| 11. Blocking question | Awaiting-input tool gate | Native child tool hook |
| 12. Changed approved file | Target comparison | Local sockets; fixture owner answer |
| 13. Two corrections and outage | Current-intent acceptance | Local sockets; every obligation closed as specified |
| 14. Unsupported platform | Platform allow-list | Platform routing and front continuation |

The fake engine's count and stream are test doubles. Scenario 2 also terminates
a real temporary subprocess, but only the clone test proves server resource
release. No socket restriction is bypassed. Tests marked `hands_sockets` are
skipped when local socket pairs are denied; the integrator must rerun them where
asyncio and aiosqlite can wake their event loops normally.

Scenarios that need a unit not merged yet xfail only while it is absent, with
the unit named in the reason: scenario 10 while `front.register` is the inert
step 0 stub (OC3), scenarios 3 and 4 while `colonymind.hands.evidence` is absent
(OC4: the check runner's probe in the digest, real E3 evaluation and one derived
autobiography entry). The moment the unit lands each runs for real and must pass;
a fixture never grants itself E3. Scenario 10 delivers the owner's message the
way Hermes does in busy mode `queue`: into the adapter's pending slot, with no
`pre_gateway_dispatch` at arrival (that hook runs only when the queued event is
processed). OC3's rebuild extends the stimulus with the turn hooks its arrival
seam arms on. Scenario 13 reads the obligation rule with section 4.4: a revision
closes the task's obligation as `superseded` and opens a successor, so the chain
in creation order must end `communicated` or `cancelled`.

Scenario 10 requires `baseline`, a list of at least three finite, nonnegative
arrival-to-owner-turn-start durations in seconds. Collect the same owner message
with no review in flight, in the same run on the same clone, separately for each
guard variant. Both `baseline` and `elapsed` end at the front's `pre_llm_call`
TURN START hook. The owner must start next, with no intervening turn, the review
must be redirected or its pending request dropped, and `elapsed` must be at most
`median(baseline) + 1.5` seconds
(rounded to milliseconds, as in the gate kit). Missing or invalid samples are
instrumentation errors and cannot count as mutation kills. The local fixture
measures its fake gateway hooks; the `hands_clone` test requires the integrator's
independent Hermes measurements.

`redirected` records actual native redirect acceptance. A dropped review request
can instead supply `request_dropped: true`; the optional field must be boolean
when present and defaults to false for existing drivers. The integrator must
observe the drop receipt from turn state or its metric and verify zero review
transport sends. A drop must not be reported as a native redirect. With neither
receipt, the guard-off case still fails `owner_first`; a drop does not relax the
owner-next or turn-start timing requirements.

## Clone replay

The package entry point is a small wrapper around `hands_scenarios.replay`.
The same runner can be invoked directly:

```sh
PYTHONPATH=sidecar/tests python -m hands_scenarios.replay \
  --stage 1 --report /tmp/hands-scenario-report.json -- \
  /path/to/isolated-clone-driver
```

The integrator supplies the clone-specific executable because the repository
has no generic gateway restart, engine-control or platform transport API. The
runner sends each catalog action sequence over stdin as JSON, in a fresh driver
process, once with its guard on and once disabled. It never substitutes fake
execution for clone evidence. The package runner alone is not a completed clone
driver; clone acceptance remains pending until that adapter is supplied and run.

Request fields include `protocol: hands-scenarios/1`, a fresh `run_id`, `scenario`,
`stage`, `steps`, `guard`, `assertion`, `guard_disabled`, `max_running` and
`max_concurrent_children`. The driver must create isolated fixture state for each
request, execute every step against the clone, disable only the named guard for
a mutation, capture real observations, and restore changes before returning.
The stdout response echoes protocol, run id, scenario, stage and guard state,
and supplies `target: rc-clone`, `isolated: true`, nonempty `evidence` artifact
references, and `observed` with the exact fields consumed by `catalog.check`.
The driver attests those artifacts' origin; this runner validates observations,
not a self-reported `passed` boolean. Return a nonzero exit code on setup,
instrumentation or cleanup failure. Do not emit credentials or conversation text.

The default per-case timeout is 300 seconds, a runner assumption adjustable with
`--timeout`. A timeout, invalid report, missing observation, wrong run, wrong
stage, surviving mutation or partial selection cannot pass the stage gate.
The runner records observations and stdout hashes in its JSON report. An overall
pass requires all fourteen baselines and all fourteen named mutation kills.

To collect the individual live tests, set `HANDS_CLONE_COMMAND` to a JSON argv
array for that driver and run `test_clone.py`. With no driver configured those
fourteen tests explicitly skip for the integrator.

Launch fixtures admit WhatsApp and SMS. Scenario 14 keeps RCS off until its
completion route is qualified, and always refuses API, voice and unknown
platforms. Any separately enabled RCS route must target only the owner's own
thread, only completed assistant work, and the exact originating conversation.
Monitoring uses the home channel, with 30 routine and 20 failure or waiting
notices daily. Hands retain shell/code access with budgets; limits are best
effort. Money, irreversible deletion, credential changes, bulk messages,
publishing and owner-account effects require the owner's yes. Clone tests use
only isolated fixture artifacts and a fixture owner identity.
