"""Design 3.4: the plugin's spool is ``hands-spool`` in the instance directory, beside ``hands.db``."""
import importlib
import sys
from types import ModuleType, SimpleNamespace

from hands_fake_hermes import FakeContext, FakeHermes, load_hands


def test_spool_lives_in_the_instance_directory(tmp_path, monkeypatch):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    native = FakeHermes()
    monkeypatch.setattr(dispatch, "NativeHermes", lambda: native)
    adapter = ModuleType("colonymind_hermes")
    adapter.__path__ = []
    client = ModuleType("colonymind_hermes.client")
    client.load_settings = lambda: SimpleNamespace(home=tmp_path, sidecar_url="http://127.0.0.1:9", api_key="")
    client.read_yaml = lambda path: {}
    monkeypatch.setitem(sys.modules, "colonymind_hermes", adapter)
    monkeypatch.setitem(sys.modules, "colonymind_hermes.client", client)
    monkeypatch.setitem(sys.modules, "tools.delegate_tool", native.delegate_module)
    monkeypatch.setattr(dispatch, "_RUNTIME", None)
    dispatch.register(FakeContext())
    runtime = dispatch.get_runtime()
    try:
        assert runtime is not None
        assert runtime.spool_dir == tmp_path / "hands-spool"
        assert not (tmp_path / "data").exists()
    finally:
        runtime.close()
