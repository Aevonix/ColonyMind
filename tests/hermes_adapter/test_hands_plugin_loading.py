"""Optional adapter loading must leave stock Hermes alone and keep real faults loud."""

import logging
import sys
from types import SimpleNamespace

import pytest

from hands_fake_hermes import FakeContext, load_hands


def test_missing_adapter_registers_nothing_and_logs_once_across_profiles(monkeypatch, caplog):
    packages = [load_hands(), load_hands()]
    monkeypatch.setitem(sys.modules, "colonymind_hermes", None)
    monkeypatch.setattr(packages[0].logger, "_adapter_absence_logged", False, raising=False)
    caplog.set_level(logging.INFO, logger="colonymind-hands")
    for package in packages:
        context = FakeContext()
        monkeypatch.setattr(package, "_refuse_delegation", lambda *args: pytest.fail("stock delegation changed"))
        package.register(context)
        package.register(context)
        assert package.ADAPTER_ABSENT is True
        assert not any((context.hooks, context.middleware, context.tools, context.commands, context.unloads))
        assert package.get_route_resolver() is None
        assert package.__name__ + ".dispatch" not in sys.modules
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.INFO, "hands need the ColonyMind adapter; not loaded in this process")]


def test_present_adapter_keeps_all_registrations_in_order_and_clears_skip(monkeypatch):
    package, context, calls = load_hands(), FakeContext(), []
    package.ADAPTER_ABSENT = True

    def load(name, **kwargs):
        if name == "colonymind_hermes":
            return SimpleNamespace()
        assert kwargs == {"package": package.__name__}
        return SimpleNamespace(register=lambda ctx: calls.append((name, ctx)),
                               activate=lambda ctx: calls.append(('activate', ctx)))

    monkeypatch.setattr(package, "import_module", load)
    package.register(context)
    assert package.ADAPTER_ABSENT is False
    assert calls == [("." + name, context) for name in package.MODULES] + [('activate', context)]


@pytest.mark.parametrize("error", [
    ModuleNotFoundError("adapter dependency missing", name="adapter_dependency"),
    ImportError("adapter import broken"),
    RuntimeError("adapter initialization broken"),
])
def test_broken_adapter_import_is_loud_and_refuses_delegation(monkeypatch, error):
    package, refused = load_hands(), []

    def broken(name, **kwargs):
        raise error

    monkeypatch.setattr(package, "import_module", broken)
    monkeypatch.setattr(package, "_refuse_delegation", lambda *args: refused.append(True))
    with pytest.raises(type(error)) as raised:
        package.register(FakeContext())
    assert raised.value is error
    assert package.ADAPTER_ABSENT is False
    assert refused == [True]
