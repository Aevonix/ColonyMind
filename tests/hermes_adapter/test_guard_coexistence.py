"""Contract-only Companion hook: either plugin's veto survives the real dispatcher.

The hook below is a verdict fixture, not a Keychain detector or implementation.
"""
import pytest

from guard_offline import offline_home, probe


@pytest.mark.parametrize("companion_first", [True, False])
@pytest.mark.parametrize("guard_blocks,companion_action", [
    (False, "block"), (True, "allow"), (True, "block"),
    (False, "allow"), (True, "approve"),
])
def test_either_plugin_veto_wins_in_both_orders(offline_home, companion_first, guard_blocks, companion_action):
    result = probe('''
from hermes_cli.plugins import PluginContext, PluginManifest, LoadedPlugin, resolve_pre_tool_block
from gateway.session_context import set_session_vars, clear_session_vars
calls = []
def companion_tripwire_contract(**kwargs):
    calls.append("companion")
    return {"action": %r, "message": "contract fixture verdict"}
manifest = PluginManifest(name="companion-tripwire-contract")
manager._plugins[manifest.name] = LoadedPlugin(manifest=manifest, enabled=True)
context = PluginContext(manifest, manager)
context.register_hook("pre_tool_call", companion_tripwire_contract)
# Install the requested registration order without replacing the dispatcher or
# either ColonyMind callback. All callbacks were registered through real contexts.
callbacks = manager._hooks["pre_tool_call"]
if %r:
    callbacks.remove(companion_tripwire_contract)
    callbacks.insert(0, companion_tripwire_contract)
before = callbacks.index(companion_tripwire_contract) < next(
    index for index, callback in enumerate(callbacks) if getattr(callback, "__module__", "") == "colonymind_hermes.guard")
sender = "2003" if %r else "1001"
tokens = set_session_vars(platform="telegram", user_id=sender, chat_id="test-chat", chat_type="dm")
try:
    invoke_hook("pre_llm_call", session_id="test-session", platform="telegram", sender_id=sender,
                turn_id="test-turn", user_message="synthetic fixture", conversation_history=[])
    # No tool executes: this is the exact dispatcher boundary used before dispatch.
    blocked = resolve_pre_tool_block("terminal", {"command": "public fixture"}, session_id="test-session",
                                     tool_call_id="test-call") is not None
finally:
    clear_session_vars(tokens)
emit(blocked=blocked, companion_called=bool(calls), companion_first=before)
''' % (companion_action, companion_first, guard_blocks), offline_home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"blocked": guard_blocks or companion_action == "block", "companion_called": True,
                      "companion_first": companion_first}


@pytest.mark.parametrize("companion_first", [True, False])
def test_instance_key_veto_also_survives_companion_allow(offline_home, companion_first):
    result = probe('''
from pathlib import Path
from hermes_cli.plugins import PluginContext, PluginManifest, LoadedPlugin, resolve_pre_tool_block
manifest = PluginManifest(name="companion-tripwire-contract")
manager._plugins[manifest.name] = LoadedPlugin(manifest=manifest, enabled=True)
def companion_allow(**kwargs):
    return {"action": "allow"}
PluginContext(manifest, manager).register_hook("pre_tool_call", companion_allow)
if %r:
    manager._hooks["pre_tool_call"].remove(companion_allow)
    manager._hooks["pre_tool_call"].insert(0, companion_allow)
guard_hook = next(callback for callback in manager._hooks["pre_tool_call"]
                  if getattr(callback, "__module__", "") == "colonymind_hermes.guard")
guard_hook.__self__.sessions.inherit("test-child", "test-parent", standing="mind")
key = guard_hook.__self__.settings.key_file
blocked = resolve_pre_tool_block("read_file", {"path": str(key)}, session_id="test-child") is not None
emit(blocked=blocked)
''' % companion_first, offline_home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"blocked": True}
