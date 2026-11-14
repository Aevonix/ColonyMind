"""Registration stays bounded even when the first controller maintenance hangs."""
import importlib
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from hands_fake_hermes import FakeContext, FakeController, FakeHermes, Parent, load_hands
from hermes_source import HERMES_ROOT, assert_supported_source


@pytest.fixture
def staged(tmp_path):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native = FakeHermes()
    runtime = dispatch.Runtime(config={'platforms':['sms']}, client=FakeController(),
                               spool_dir=tmp_path / 'spool', native=native,
                               owner_check=lambda sid: sid.startswith('owner'))
    native.runtime = runtime
    context = FakeContext()
    context.hands_runtime = runtime
    yield package, dispatch, runtime, native, context
    runtime.close()
    front = importlib.import_module(package.__name__ + '.front')
    owned = getattr(context, 'hands_front', None)
    if owned is not None:
        owned.close()


def test_slow_maintenance_does_not_delay_plugin_registration(staged, monkeypatch):
    package, dispatch, runtime, native, context = staged
    entered, release, registered = threading.Event(), threading.Event(), threading.Event()
    original_install = runtime.install
    errors = []

    def blocked():
        entered.set()
        release.wait(2)

    def install():
        assert len(context.hooks['pre_tool_call']) >= 4
        assert runtime.close in context.unloads
        original_install()

    def register():
        try:
            package.register(context)
        except BaseException as error:
            errors.append(error)
        finally:
            registered.set()

    monkeypatch.setattr(runtime, 'maintain', blocked)
    monkeypatch.setattr(runtime, 'install', install)
    worker = threading.Thread(target=register)
    worker.start()
    try:
        assert registered.wait(1), 'sidecar maintenance delayed registration'
        assert not errors
        assert entered.wait(1)
        assert not release.is_set()
        assert getattr(native.delegate_module.delegate_task, '_hands_runtime', None) is runtime
    finally:
        release.set()
        worker.join(2)
        for unload in reversed(context.unloads):
            unload()


def test_abandoned_load_never_leaves_unguarded_delegate_wrapper(staged, monkeypatch):
    package, dispatch, runtime, native, context = staged
    revision = assert_supported_source()
    monkeypatch.syspath_prepend(str(HERMES_ROOT))
    monkeypatch.setitem(sys.modules, 'tools.delegate_tool', native.delegate_module)
    guard = importlib.import_module(package.__name__ + '.budgets')
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = guard.register
    context._abandon_load = lambda: setattr(context, '_load_abandoned', True)

    def blocked(ctx):
        entered.set()
        release.wait(2)
        original(ctx)

    def register():
        try:
            package.register(context)
        finally:
            finished.set()

    monkeypatch.setattr(guard, 'register', blocked)
    if revision in {'S', 'S217'}:
        loader = importlib.import_module('hermes_cli.plugins_loader')
        monkeypatch.setattr(loader, '_resolve_plugin_load_timeout', lambda: .02)
        with pytest.raises(loader.PluginLoadTimeout):
            loader.run_with_load_deadline('hands-test', context, register)
    else:
        worker = threading.Thread(target=register)
        worker.start()
        assert entered.wait(1)
        assert not finished.wait(.02)
        context._abandon_load()
    release.set()
    assert finished.wait(2)
    assert context._load_abandoned
    answer = json.loads(native.delegate_module.delegate_task(goal='refused', parent_agent=Parent()))
    assert answer.get('error') and not native.calls
    assert not getattr(native.delegate_module.delegate_task, '_hands_runtime', None)
    # Reload unwraps only the owned refusal and can install a guarded successor.
    successor = type(runtime)(config=runtime.config, client=runtime.client,
        spool_dir=runtime.spool_dir / 'reload', native=native, owner_check=runtime.owner_check)
    new_context = FakeContext()
    new_context.hands_runtime = successor
    monkeypatch.setattr(guard, 'register', original)
    try:
        package.register(new_context)
        installed = native.delegate_module.delegate_task
        assert getattr(installed, '_hands_runtime', None) is successor
        dispatch.activate(context)
        assert native.delegate_module.delegate_task is installed
        for unload in reversed(context.unloads):
            unload()
        assert native.delegate_module.delegate_task is installed
    finally:
        for unload in reversed(new_context.unloads):
            unload()
        successor.close()


def test_abandonment_after_install_refuses_every_dispatch_path(staged):
    package, dispatch, runtime, native, context = staged
    package.register(context)
    context._load_abandoned = True
    assert json.loads(native.delegate_module.delegate_task(goal='check', parent_agent=Parent()))['error']
    assert runtime.dispatch_tasks([{'goal':'check'}], parent=Parent())['error']
    assert native.async_module.dispatch_async_delegation(goal='check')['error']
    assert not native.calls
    for unload in reversed(context.unloads):
        unload()


def test_reload_before_abandoned_loader_returns_gets_a_new_runtime(staged, monkeypatch):
    package, dispatch, runtime, native, context = staged
    monkeypatch.setitem(sys.modules, 'tools.delegate_tool', native.delegate_module)
    dispatch.register(context)
    context._load_abandoned = True
    successor = type(runtime)(config=runtime.config, client=runtime.client,
        spool_dir=runtime.spool_dir / 'reload', native=native, owner_check=runtime.owner_check)
    from colonymind_hermes import client
    monkeypatch.setattr(client, 'load_settings', lambda:SimpleNamespace(home=runtime.spool_dir))
    monkeypatch.setattr(client, 'read_yaml', lambda path:{})
    monkeypatch.setattr(dispatch, 'Runtime', lambda **kwargs:successor)
    new_context = FakeContext()
    try:
        dispatch.register(new_context)
        assert dispatch.get_runtime(new_context) is successor
        assert runtime._closed.is_set() and not runtime.available()
        assert context.hands_runtime is runtime
        assert successor.available()
        dispatch.activate(new_context)
        current = native.delegate_module.delegate_task
        dispatch.activate(context)
        assert native.delegate_module.delegate_task is current
        assert getattr(current, '_hands_runtime', None) is successor
        # A late exception from the abandoned worker uses the same ownership
        # check as its activation path, so it cannot refuse its successor.
        package._refuse_delegation(context)
        assert native.delegate_module.delegate_task is current
    finally:
        successor.close()
