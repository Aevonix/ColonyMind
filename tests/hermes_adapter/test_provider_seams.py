"""The memory provider's stock seams: availability without the network, why it is unavailable, what a Hermes
backup must carry, and the owner-only recall indicator."""

import json

from conftest import probe

PRELUDE = '''
import json, os, pathlib, socket
def emit(**values):
    print("@@RESULT@@" + json.dumps(values, default=str), flush=True)
def no_network(*args, **kwargs):
    raise AssertionError("the availability check opened a connection")
from plugins.memory import load_memory_provider
provider = load_memory_provider("colonymind-memory")
assert provider is not None, "provider did not load"
import colonymind_memory, yaml
ADAPTER_VERSION = yaml.safe_load(pathlib.Path(colonymind_memory.__file__).with_name("plugin.yaml").read_text())["version"]
'''

AVAILABILITY = '''
socket.socket.connect = no_network
socket.create_connection = no_network
emit(available=provider.is_available(), reason=provider.unavailable_reason(), adapter=ADAPTER_VERSION)
'''


def _install_state(home, **fields):
    (home.instance / "hermes-install.json").write_text(json.dumps(fields))


def test_available_when_configured_even_with_the_sidecar_down(home, sidecar):
    """A sidecar outage stays a per-request report: deactivating the provider would stop captures."""
    home.write_config(**{**home.config, "plugins": {**home.config["plugins"], "colonymind": {
        "sidecar_url": "http://127.0.0.1:9", "key_file": str(home.key_file)}}})
    result = probe(AVAILABILITY, home, prelude=PRELUDE)
    assert result == {"available": True, "reason": "", "adapter": result["adapter"]}


def test_unavailable_without_its_key_file_and_says_so_without_the_network(home, sidecar):
    home.key_file.unlink()
    result = probe(AVAILABILITY, home, prelude=PRELUDE)
    assert result["available"] is False
    assert result["reason"] == f"the key file {home.key_file} is missing: run 'colonymind init' (it writes one)"


def test_unavailable_with_an_unreadable_key_file(home, sidecar):
    home.key_file.chmod(0o000)
    try:
        result = probe(AVAILABILITY, home, prelude=PRELUDE)
    finally:
        home.key_file.chmod(0o600)
    assert result["available"] is False
    assert result["reason"].startswith(f"the key file {home.key_file} cannot be read (PermissionError)")


def test_unavailable_without_a_sidecar_url(home, sidecar):
    home.write_config(**{**home.config, "plugins": {"enabled": ["colonymind"], "hook_callback_timeout": 0}})
    result = probe(AVAILABILITY, home, prelude=PRELUDE)
    assert result["available"] is False
    assert result["reason"].startswith("no sidecar URL is set (plugins.colonymind.sidecar_url in Hermes' config)")
    assert "colonymind init" in result["reason"]


def test_available_when_the_adapter_and_the_sidecar_init_recorded_disagree(home, sidecar):
    """What init once recorded is not the sidecar that runs now, so it never switches memory off."""
    _install_state(home, sidecar_version="0.0.1", adapter_version="0.0.1")
    result = probe(AVAILABILITY, home, prelude=PRELUDE)
    assert result["adapter"] != "0.0.1"
    assert result == {"available": True, "reason": "", "adapter": result["adapter"]}


def test_backup_paths_name_the_instance_directory_through_stock_backup(home, sidecar):
    result = probe('''
socket.socket.connect = no_network
from hermes_cli.backup import _collect_memory_provider_external_paths
emit(declared=provider.backup_paths(), collected=[str(path) for path in _collect_memory_provider_external_paths()])
''', home, prelude=PRELUDE)
    assert result["declared"] == [str(home.instance.resolve())]
    assert result["collected"] == [str(home.instance.resolve())]


RECALL = '''
from agent.memory_manager import MemoryManager
from gateway.session_context import set_session_vars, clear_session_vars
provider.initialize("session-1", hermes_home=os.environ["HERMES_HOME"], platform=PLATFORM)
manager = MemoryManager()
manager.add_provider(provider)
seen = {}
for label, binding in TURNS:
    tokens = set_session_vars(**binding) if binding else None
    try:
        context = manager.prefetch_all("where does the ferry leave from", session_id="session-1")
        seen[label] = {"indicator": manager.describe_recall(), "context": bool(context)}
    finally:
        if tokens is not None:
            clear_session_vars(tokens)
emit(**seen)
'''


def _recalling(sidecar):
    """Every assembly carries a memory section citing three quotations."""
    def assemble(body):
        refs = [{"source_id": f"src-{n}", "source_version": "a" * 64} for n in range(3)]
        return {"sections": [{"id": "shared", "title": "Shared", "body": "shared facts", "priority": 50},
                             {"id": "colonymind-memory", "title": "Relevant Memories", "body": "three quotations",
                              "priority": 90, "citations": refs}]}
    sidecar.assemble = assemble


def test_recall_status_says_n_to_the_owner_and_nothing_in_a_contacts_chat(home, sidecar):
    _recalling(sidecar)
    turns = [
        ("owner_dm", {"platform": "telegram", "user_id": "1001", "chat_id": "1001", "chat_type": "dm",
                      "session_id": "session-1"}),
        ("contact_dm", {"platform": "telegram", "user_id": "2003", "chat_id": "2003", "chat_type": "dm",
                        "session_id": "session-1"}),
        ("owner_in_a_group", {"platform": "telegram", "user_id": "1001", "chat_id": "-100500", "chat_type": "group",
                              "session_id": "session-1"}),
        ("owner_again", {"platform": "telegram", "user_id": "1001", "chat_id": "1001", "chat_type": "dm",
                         "session_id": "session-1"}),
    ]
    result = probe(f"PLATFORM = 'telegram'\nTURNS = {turns!r}\n" + RECALL, home, prelude=PRELUDE)
    assert "ColonyMind" in result["owner_dm"]["indicator"] and "recalled 3 memories" in result["owner_dm"]["indicator"]
    assert result["contact_dm"] == {"indicator": "", "context": True}          # recalled, never announced
    assert result["owner_in_a_group"]["indicator"] == ""                        # others read that chat
    assert "recalled 3 memories" in result["owner_again"]["indicator"]


def test_recall_status_on_the_owners_terminal(home, sidecar):
    _recalling(sidecar)
    result = probe("PLATFORM = 'cli'\nTURNS = [('cli', None)]\n" + RECALL, home, prelude=PRELUDE)
    assert "recalled 3 memories" in result["cli"]["indicator"]
