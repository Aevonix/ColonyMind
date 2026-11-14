"""Clock decisions follow the host interface without a standalone rule."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

PROVIDER = Path(__file__).resolve().parents[2] / "plugins" / "colonymind-memory" / "provider.py"


def _load():
    spec = importlib.util.spec_from_file_location("memory_host_predicate_test", PROVIDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_decides_even_for_prompts_outside_the_old_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "instance"))
    calls = []
    host = types.ModuleType("agent.memory_provider")

    def predicate(text):
        calls.append(text)
        return text == "host-defined skip"

    host.is_trivial_prompt = predicate
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.memory_provider", host)
    module = _load()
    assert module.is_trivial_prompt("host-defined skip") is True
    assert module.is_trivial_prompt("ok") is False
    assert calls == ["host-defined skip", "ok"]
    provider = module.ColonyMindMemoryProvider(config={})
    assert provider.turn_clock(message="host-defined skip").startswith("the assistant's time: ")
    assert provider.turn_clock(message="ok") == ""
    assert calls[-2:] == ["host-defined skip", "ok"]


def test_missing_host_interface_has_no_replacement_predicate(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.memory_provider", None)
    module = _load()
    with pytest.raises(ImportError):
        module.is_trivial_prompt("ok")
    assert not hasattr(module, "_standalone_is_trivial_prompt")
    assert not hasattr(module, "_TRIVIAL_PROMPT_RE")
