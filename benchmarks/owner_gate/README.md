# owner-gate-1

The owner gate (docs/OWNER-GATE.md) measured on the messages it exists for, with no model. Twelve scenarios
over the served mind (real stores, a fake clock, quiet hours 22:00-07:00), each run with
`mind.faculties.owner_gate` off (today's path: every approved owner message goes out, and an outside script
texts every alert) and on. Counted: the owner's interruptions.

```sh
python benchmarks/owner_gate/measure.py --out summary.json
```

| Group | Scenarios | Right sends | Off | On | Right, off | Right, on |
|---|---:|---:|---:|---:|---:|---:|
| warranted | 6 | 7 | 7 | 7 | 6/6 | 6/6 |
| control | 6 | 3 | 12 | 3 | 0/6 | 6/6 |

`reference-results.json` is this run, per scenario with the gate's verdicts; `sidecar/tests/test_owner_gate_measure.py`
checks that the code still gives it. The second opinion is a scripted answer in the two night scenarios that
need one: the measurement is of the gate's mechanics, not of a model's judgment.
