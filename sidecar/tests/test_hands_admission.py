"""Contract chains and engine metrics are independent of executor slot counts."""

from colonymind.hands import contracts, health
import os
import pytest
from test_hands_controller import c, contract, plan


def test_contract_chain_drops_fallback_missing_modality():
    configured = {"vision": contract(modalities=["text", "image"]), "text": contract()}
    valid, dropped = contracts.validate_chain(["vision", "text"], configured,
                                               needs=["vision"], kind="media", input_tokens=100)
    assert valid == ["vision"] and "text" in dropped
    assert contracts.validate_chain(["missing"], configured, needs=[], kind="deep", input_tokens=10)[0] == []


def test_metrics_reader_rejects_partial_nan_and_aggregates_labels():
    metrics = ('vllm:num_requests_running{model_name="example"} 2\n'
               'vllm:num_requests_waiting{model_name="example"} 0\n'
               'vllm:kv_cache_usage_perc{model_name="example"} 0.4\n')
    assert health.parse_metrics(metrics) == {"running": 2.0, "waiting": 0.0, "kv": 0.4}
    for bad in (metrics.replace("0.4", "NaN"), metrics.splitlines()[0]):
        assert health.parse_metrics(bad) is None


async def test_engine_truth_and_unreadable_owner_only(c):
    c.health.record("family-a", running=0, waiting=1, kv=0.1)
    assert (await plan(c))["reason"] == "engine_busy"
    c.health.record("family-a", running=0, waiting=0, kv=0.8)
    assert (await plan(c, "kv"))["reason"] == "engine_busy"
    c.health.invalidate("family-a")
    assert (await plan(c, "own", origin="own"))["admit"] != "start"
    assert (await plan(c, "owner"))["admit"] == "start"
    assert (await plan(c, "owner-2"))["admit"] != "start"


@pytest.mark.skipif(not os.environ.get("HANDS_LIVE_METRICS_URL"),
                    reason="Integrator: engine metrics test needs sockets and a live metrics endpoint")
async def test_live_engine_metrics_reader():
    reader = health.EngineHealth({"poll_seconds": 5,
                                  "metrics_urls": {"engine": os.environ["HANDS_LIVE_METRICS_URL"]}})
    await reader.refresh()
    assert "engine" in reader.readings
