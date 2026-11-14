"""Integrator fixes: a failed module registration refuses delegation, and advertised limits come from config."""
from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from hands_fake_hermes import FakeContext, FakeController, FakeHermes, Parent, load_hands


def test_a_module_failing_to_register_leaves_delegation_refused(tmp_path, monkeypatch):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    commands = importlib.import_module(package.__name__ + ".commands")
    native = FakeHermes()
    monkeypatch.setattr(dispatch, "NativeHermes", lambda: native)
    adapter = ModuleType("colonymind_hermes")
    adapter.__path__ = []
    client = ModuleType("colonymind_hermes.client")
    client.load_settings = lambda: SimpleNamespace(home=tmp_path, sidecar_url="http://127.0.0.1:9", api_key="k")
    client.read_yaml = lambda path: {"hands": {"platforms": ["sms"]}}
    monkeypatch.setitem(sys.modules, "colonymind_hermes", adapter)
    monkeypatch.setitem(sys.modules, "colonymind_hermes.client", client)
    monkeypatch.setitem(sys.modules, "tools.delegate_tool", native.delegate_module)

    def broken(ctx):
        raise TypeError("host rejected the command")

    monkeypatch.setattr(commands, "register", broken)
    with pytest.raises(TypeError):
        package.register(FakeContext())
    result = json.loads(native.delegate_module.delegate_task(goal="check", parent_agent=Parent()))
    assert result["error"] == "hands runtime unavailable"
    assert not native.calls


def test_notice_caps_and_timings_follow_hands_notices(tmp_path):
    package = load_hands()
    notices = importlib.import_module(package.__name__ + ".notices")
    emitter = notices.NoticeEmitter(client=None, emit=None, sink=None, ledger_path=tmp_path / "n.sqlite3",
                                    settings=lambda: {"routine_per_day": 2, "actionable_per_day": 1,
                                                      "merge_seconds": 5, "overflow_seconds": 60})
    assert emitter._maximum("routine") == 2 and emitter._maximum("actionable") == 1
    assert emitter.limits()["merge_seconds"] == 5 and emitter.limits()["overflow_seconds"] == 60
    fallback = notices.NoticeEmitter(client=None, emit=None, sink=None, ledger_path=tmp_path / "m.sqlite3",
                                     settings=lambda: {"routine_per_day": "many"})
    assert fallback._maximum("routine") == 30 and fallback._maximum("actionable") == 20


def test_degraded_threshold_and_window_follow_hands_degraded():
    package = load_hands()
    degraded = importlib.import_module(package.__name__ + ".degraded")
    mode = degraded.DegradedMode(settings=lambda: {"controller_failures": 3, "failure_window_seconds": 5})
    mode.failure(0)
    mode.failure(1)
    assert not mode.is_degraded
    mode.failure(10)
    assert not mode.is_degraded
    mode.failure(11)
    mode.failure(12)
    assert mode.is_degraded


def test_reconcile_sends_only_hands_with_an_executor(tmp_path):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    budgets = importlib.import_module(package.__name__ + ".budgets")
    reconcile = importlib.import_module(package.__name__ + ".reconcile")
    native, client = FakeHermes(), FakeController()
    runtime = dispatch.Runtime(config={}, client=client, spool_dir=tmp_path / "spool", native=native,
                               owner_check=lambda sid: sid.startswith("owner"))
    try:
        runtime.registry.add(budgets.HandState(hand_id="h-" + "a" * 12, status="reserved", epoch=runtime.epoch))
        runtime.registry.add(budgets.HandState(hand_id="h-" + "b" * 12, status="refused", epoch=runtime.epoch))
        live = budgets.HandState(hand_id="h-" + "c" * 12, status="running", epoch=runtime.epoch,
                                 delegation_id="delegation-1", subagent_id="subagent-1")
        runtime.registry.add(live)
        reconcile.reconcile(runtime, lambda ident: None)
        sent = [body for path, body, _ in client.writes if path == "/v1/hands/reconcile"]
        assert [row["hand_id"] for row in sent[0]["live"]] == [live.hand_id]
        assert all(row["executor_ref"] for row in sent[0]["live"])
    finally:
        runtime.close()
