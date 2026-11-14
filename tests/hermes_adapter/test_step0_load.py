"""Enabled plugin loading without a socket fixture or installed-checkout ambiguity."""
import json
import os
from pathlib import Path
import subprocess
import sys

from conftest import ROOT, build_home
from hermes_source import assert_supported_source


def test_selected_revision_loads_enabled_worktree_plugin(tmp_path):
    assert_supported_source()
    home = build_home(tmp_path, 9, 'http://127.0.0.1:9')
    env = dict(os.environ)
    env.update(HERMES_HOME=str(home.hermes), COLONYMIND_HOME=str(home.instance),
               COLONYMIND_TEST_HOME=str(home.root), COLONYMIND_BODY_THREAD='0', HERMES_DISABLE_LAZY_INSTALLS='1',
               PYTHON_DOTENV_DISABLED='1', COLONYMIND_SKIP_DOTENV='1')
    code = '''
import json
from hermes_cli.plugins import get_plugin_manager, _reset_plugin_managers_for_tests, VALID_HOOKS
_reset_plugin_managers_for_tests()
manager = get_plugin_manager()
manager.discover_and_load(force=True)
loaded = manager._plugins['colonymind']
assert loaded.enabled, loaded.error
import colonymind_hermes
from agent.turn_facade import TurnFacadeMixin
from colonymind_hermes import native_settlement
callbacks = {name: [cb for cb in hooks if cb.__module__.startswith('colonymind_hermes')]
             for name, hooks in manager._hooks.items()}
assert all(name in VALID_HOOKS for name, hooks in callbacks.items() if hooks)
if 'on_native_turn_settled' in VALID_HOOKS:
    assert len(callbacks['on_native_turn_settled']) == 2
else:
    assert not callbacks.get('on_native_turn_settled')
    assert getattr(TurnFacadeMixin.run_conversation, native_settlement._MARKER, None)
print('RESULT:' + json.dumps({'adapter': colonymind_hermes.__file__, 'enabled': loaded.enabled}))
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith('RESULT:')))
    assert payload['enabled']
    assert Path(payload['adapter']).read_bytes() == (ROOT / 'plugins/hermes-plugin/__init__.py').read_bytes()


def test_selected_revision_loads_memory_and_enabled_gates_without_io(tmp_path):
    assert_supported_source()
    home = build_home(tmp_path, 9, 'http://127.0.0.1:9')
    plugins = dict(home.config['plugins'], enabled=['colonymind', 'colonymind-gates'])
    plugins['colonymind-gates'] = {'triage': {'enabled': True}, 'screen': {'enabled': True}}
    home.write_config(plugins=plugins)
    env = dict(os.environ, HERMES_HOME=str(home.hermes), COLONYMIND_HOME=str(home.instance),
               COLONYMIND_BODY_THREAD='0', HERMES_DISABLE_LAZY_INSTALLS='1',
               PYTHON_DOTENV_DISABLED='1', COLONYMIND_SKIP_DOTENV='1')
    code = '''
import sys
from pathlib import Path
def offline(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise AssertionError('registration must not use sockets')
sys.addaudithook(offline)
from hermes_cli.plugins import discover_plugins, get_plugin_manager
from plugins.memory import load_memory_provider
discover_plugins()
manager = get_plugin_manager()
for name in ('colonymind', 'colonymind-gates'):
    loaded = manager._plugins[name]
    assert loaded.enabled and not loaded.error, (name, loaded.error)
assert set(manager._plugins['colonymind-gates'].hooks_registered) == {
    'pre_gateway_dispatch', 'transform_tool_result'}
from colonymind_memory import ColonyMindMemoryProvider
import colonymind_memory
provider = load_memory_provider('colonymind-memory')
assert isinstance(provider, ColonyMindMemoryProvider)
assert provider.is_available(), provider.unavailable_reason()
assert not provider.unavailable_reason()
assert any(Path(cb.__code__.co_filename).resolve() == Path(colonymind_memory.__file__).resolve()
           for cb in manager.iter_hook_callbacks('pre_llm_call') if hasattr(cb, '__code__'))
manager.unload()
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
