# Hardware-qualified latency gates

Local runs enforce each test's unchanged wall-time bar. On GitHub Actions
(`GITHUB_ACTIONS=true`), tests report measurements without failing on latency.
The `hardware_latency` marker identifies this policy; tests use the fixture of the
same name for the measured assertion. Disk I/O and other functional failures always fail.

The p99 < 1 ms hook tests are in `tests/hermes_adapter/test_work_capture.py`:

- `test_new_metadata_hooks_are_sync_extensible_enqueue_only_under_one_ms`
- `test_maximum_resource_patch_hook_stays_under_one_ms`

The sidecar test
`tests/test_plasticity_service_health_stall.py::test_pass_runs_off_the_event_loop_while_a_large_sample_file_is_processed`
uses the same policy with its maximum event-loop gap < 250 ms bar. It always
checks that collection runs on a worker thread and that the full sample file is
processed correctly. The ticker records wall and thread CPU gaps; its minimum
20 ticks and maximum gap are enforced locally. Run this test
from `sidecar/` with the same report options. Hand-budget deadline tests use
integral synthetic timestamps and keep their functional assertions unconditional.

The sidecar's `tests/test_voice_affect_gate.py::test_voice_affect_seeded_latency_gate`
retains its nearest-rank p95 < 20 ms bar over 300 requests after 20 warmups.
Its route decision clock is controlled so runner scheduling cannot trigger the
production deadline's intentional empty fallback. Wall and thread CPU clocks
remain real; the exact nonempty line and all quality checks always apply.
`tests/test_voice_affect.py::test_late_read_discards_line` separately checks that
the production deadline discards late output.
`tests/test_voice_affect_readonly.py::test_seeded_p95_and_readonly_gate` reuses
the same measurement policy and keeps its database snapshots and SQL read-only
authorizer checks unconditional.

Run locally on an idle machine and retain the measurements:

```sh
python -m pytest tests/hermes_adapter/test_work_capture.py -m hardware_latency \
  --hardware-latency-report="$QUALIFICATION_OUTPUT/hardware-latency.json" \
  --junitxml="$QUALIFICATION_OUTPUT/hardware-latency.xml"
```

Run repeated independent batches and retain every report, including
failures. Reports include wall and thread CPU samples/p99, sample count, exact node id,
threshold, selected statistic (p99, nearest-rank p95 or maximum), enforcement mode and runtime/profile identity. The wall metric decides the
gate; CPU time helps diagnose scheduling effects. Ordinary pytest runs print each
measurement and put it in JUnit properties; `--hardware-latency-report` also saves JSON.
