"""Real numeric nightly builds, live regression rollback and closure retention."""
from datetime import timedelta

import numpy as np

from colonymind.plasticity import _io
from colonymind.plasticity.compaction import compact
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.numeric import load_revision
from test_plasticity_service_health import NOW, runtime


def test_night_one_cycle_promotes_night_two_regression_rolls_back(tmp_path):
    now = [NOW]
    service = runtime(tmp_path, now, learning_rate=.001)
    try:
        driver = service.loaded["health"].driver
        # Fourteen days at five-minute resolution, with 12 past values and rate .001.
        # These are fixture assumptions for a smooth daily cycle in standardised units.
        examples = []
        for i in range(12, 288 * 14):
            at = NOW - timedelta(seconds=300 * (288 * 14 - i))
            history = [np.sin((i - lag) * 2 * np.pi / 288) for lag in range(1, 13)]
            frame = {"at": at.isoformat(), "streams": {"value": history}, "series": "router.primary.latency_p95_ms"}
            examples.append((frame, float(np.sin(i * 2 * np.pi / 288))))
        driver.samples.add_many(examples)
        result, = service.run()
        assert result["result"] == "promoted", result
        binding = service.loaded["health"]
        first = binding.store.revisions.serving()
        driver.export_revision(first, tmp_path / "export")
        revision = load_revision(tmp_path / "export")
        np.testing.assert_array_equal(revision.weights, binding.store.revisions.state(first)["W"])
        assert driver.last_report.passed
        assert {c.name for c in driver.last_report.checks} >= {
            "retention", "specificity", "canaries", "calibration", "budget", "finite", "served_offline"}
        manifest = binding.store.current_manifests()[-1]
        assert manifest.admitted() and not (driver.sets.protected_refs() & {r.ref for r in manifest.admitted()})
        now[0] += timedelta(days=1)
        # The second night's realised targets favour persistence on the same inputs.
        comparator = binding.comparator
        for i in range(30):
            frame = {"at": (now[0] + timedelta(minutes=i)).isoformat(), "streams": {"value": [3.] * 12}}
            service.serve("health", f"night-two-{i}", frame,
                          predict=lambda pin, value: float(driver.learner.predict(pin.state, [value])[0, 0]))
        for i in range(30):
            outcome = service.outcome("health", f"night-two-{i}", observed=3., teacher="realized_value")
        assert outcome["rolled_back"]
        assert binding.store.revisions.serving().mode == "base"
        assert comparator.summary()["last_rollback"]["closure"] == first.id
        # Gate exclusions and machine samples also follow the runtime's erasure path.
        refs = [manifest.admitted()[0].ref, driver._gate_rows[0].ref]
        assert service.erase(refs)["clean"]
        assert not set(refs) & {row["ref"] for row in driver.source.rows(driver.objective)}
    finally:
        service.close()


def test_numeric_closures_compact_to_fourteen(tmp_path):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        binding = service.loaded["health"]
        for i in range(20):
            state = binding.driver.learner.base()
            state["W"][0, -1] = .001 * (i + 1)
            closure = binding.store.revisions.compose(state=_io.state_digest(state), objective="numeric",
                calibration=(f"fixture-{i}",), dependents={}, parent=binding.store.revisions.serving().id,
                manifests=(), periods=(), audience="owner", mode="stack")
            binding.controller.propose(closure, state)
            binding.controller.consider(closure.id, GateReport(closure.id, "fixture", (Check("budget", True),)))
            compact(binding.store)
        assert len(binding.store.revisions.retained()) == 14
    finally:
        service.close()
