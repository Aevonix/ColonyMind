"""owner-gate-1 (``benchmarks/owner_gate/measure.py``): with the owner gate on, no warranted send is lost and every
control sends no more than today's path, fewer in all; the frozen reference results are what the code gives."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "owner_gate"


def _module():
    spec = importlib.util.spec_from_file_location("owner_gate_measure", BENCH / "measure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_the_gate_loses_no_warranted_send_and_sends_fewer_in_the_controls(tmp_path, monkeypatch):
    measure = _module()
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", measure.OWNER)
    summary = await measure.measure(tmp_path)
    rows = summary["scenarios"]
    assert summary["warranted_sends_lost"] == 0
    assert all(row["gate"] == row["base"] == row["right"] for row in rows if row["group"] == "warranted")
    controls = [row for row in rows if row["group"] == "control"]
    assert all(row["gate"] <= row["base"] for row in controls) and all(row["gate_right"] for row in controls)
    assert summary["control"]["gate_sends"] < summary["control"]["base_sends"]
    reference = json.loads((BENCH / "reference-results.json").read_text(encoding="utf-8"))
    assert summary == reference
