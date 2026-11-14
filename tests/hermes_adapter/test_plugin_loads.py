"""The adapter loads on stock Hermes through public seams only (M1 acceptance test 7), and its one prompt
section carries the constitution, the owner and the self-narrative within Hermes' bounds (M8, seam 2)."""

import yaml

from conftest import PRELUDE, probe

RENDER = '''
from hermes_cli.plugins import render_system_prompt_sections
def section():
    sections = render_system_prompt_sections({"session_id": "s", "platform": "cli"})
    return next(s.content for s in sections if s.id == "colonymind")
'''
CONSTITUTION = "You are Agent. Your values: care. Your boundaries: never send money."
NARRATIVE_LEAD = "What you know about yourself, from your own record"

# pre_gateway_dispatch: the owner-only /mind gate (Hermes hands the command itself no sender), which also hears
# each message's own evidence and reads the owner's reaction on an ask; on_skill_lifecycle: loads of ColonyMind's own
# skills (M9); the answer route's chat ledger (answers.py): on_interim_message, transform_llm_output (an invented
# code instruction removed) and gateway_platform_event (a Telegram reaction).
STOCK_HOOKS = {"pre_llm_call", "post_llm_call", "pre_tool_call", "on_kanban_dispatch_tick", "pre_gateway_dispatch",
               "on_skill_lifecycle", "post_tool_call", "on_interim_message", "transform_llm_output",
               "gateway_platform_event", "subagent_start", "on_native_turn_settled",
               "pre_api_request", "post_api_request", "api_request_error",
               "pre_approval_request", "post_approval_response"}  # stock approval observers, returns ignored
# The sealed store's Hermes half (secrets.py, docs/SECRETS.md) adds a second pre_gateway_dispatch (a secret is
# sealed before Hermes sees it) and pre_tool_call (a send carrying a reference waits for the owner's word),
# post_tool_call (a call's private value files go when it is over), and the tool_request middleware that fills
# or hides sealed values. The mind's floor adds the other post_tool_call: what a mind run made (a cron job's id,
# a kanban task), told to the sidecar once the call returned. The answer route adds a pre_llm_call (the owner's
# message read before the model runs), a post_llm_call (the reply noted in the owner's chat) and a post_tool_call
# (the model's own send noted there).
# Busy-path ingress parity (busy_ingress.py) applies its adapter wrap from the stock hooks that see the gateway early:
# one more pre_gateway_dispatch and pre_llm_call.
# Activity adds run/tool observations; settlement uses the existing capture callback.
HOOK_COUNTS = {"pre_gateway_dispatch": 3, "pre_tool_call": 3, "post_tool_call": 5, "pre_llm_call": 4,
               "post_llm_call": 3, "on_native_turn_settled": 2, "subagent_start": 2}
MIDDLEWARE = {"tool_request": 1}
TOOLS = {"colonymind_self", "colonymind_people", "colonymind_memory_search", "colonymind_memory_forget",
         "colonymind_reminder", "colonymind_secret"}


def test_plugin_registers_only_stock_seams(home, sidecar):
    prelude = PRELUDE.replace("manager.discover_and_load(force=True)", """
import colonymind_hermes
from colonymind_hermes.client import ColonyMindClient
registration_threads = []
original_registration = colonymind_hermes.register
def observed_registration(context):
    registration_threads.append(threading.get_ident())
    return original_registration(context)
colonymind_hermes.register = observed_registration
network_threads = []
original_request = ColonyMindClient.request
def observed_request(self, method, path, **kwargs):
    network_threads.append((path, threading.get_ident(), threading.current_thread().name))
    return original_request(self, method, path, **kwargs)
ColonyMindClient.request = observed_request
manager.discover_and_load(force=True)
""")
    result = probe('''
from tools.registry import registry
def ours(callbacks):
    return [cb for cb in callbacks if "colonymind_hermes" in str(getattr(cb, "__module__", ""))]
hooks = {name: len(ours(cbs)) for name, cbs in manager._hooks.items() if ours(cbs)}
middleware = {name: len(ours(cbs)) for name, cbs in manager._middleware.items() if ours(cbs)}
emit(hooks=hooks, middleware=middleware, valid=[h for h in hooks if h not in VALID_HOOKS],
     fork_settlement='on_native_turn_settled' in VALID_HOOKS,
     registered=loaded.hooks_registered, commands=list(manager._plugin_commands),
     sections=list(manager._system_prompt_sections),
     tools=[name for name in %r if registry.get_entry(name) is not None],
     platforms=[name for name in getattr(manager, "_platforms", {}) if "colonymind" in name],
     registration_threads=registration_threads, network_threads=list(network_threads))
''' % (sorted(TOOLS),), home, prelude=prelude)
    expected = STOCK_HOOKS if result['fork_settlement'] else STOCK_HOOKS - {'on_native_turn_settled'}
    assert set(result["hooks"]) == expected, result["hooks"]
    assert result["hooks"] == {hook: HOOK_COUNTS.get(hook, 1) for hook in expected}
    assert result["valid"] == []  # no patched-hook probes: every hook is in VALID_HOOKS
    assert result["middleware"] == MIDDLEWARE
    assert "mind" in result["commands"]
    assert "colonymind" in result["sections"]
    assert set(result["tools"]) == TOOLS
    assert result["platforms"] == []
    assert result["registration_threads"]
    assert all(ident not in result["registration_threads"] for _, ident, _ in result["network_threads"])
    assert all(name == "colonymind-activity" for path, _, name in result["network_threads"]
               if path == "/v1/host/activity/snapshot")
    assert all(r["path"] in {"/v1/mind/state", "/v1/host/activity/snapshot"} for r in sidecar.requests)


def test_registration_reads_key_file_and_sidecar_url(home, sidecar):
    result = probe('''
from colonymind_hermes.client import load_settings
s = load_settings()
emit(url=s.sidecar_url, key=s.api_key, home=str(s.home), worker=s.worker_profile, outbox=str(s.outbox_path))
''', home)
    assert result["url"] == sidecar.url
    assert result["key"] == home.key_file.read_text().strip()
    assert result["home"] == str(home.instance)
    assert result["worker"] == "colonymind-act"
    assert result["outbox"].startswith(str(home.hermes))


def test_plugin_loads_without_identity_or_mind_files(home, sidecar):
    (home.instance / "identity.yaml").unlink()
    (home.instance / "colonymind.yaml").unlink()
    result = probe("emit(ok=loaded.enabled, error=loaded.error)", home)
    assert result["ok"] and not result["error"]


def test_prompt_section_carries_the_constitution_without_the_mind_routes(home, sidecar):
    """The constitution is read from identity.yaml by the plugin itself: no sidecar, no mind routes needed."""
    result = probe(RENDER + "emit(content=section())", home)
    content = result["content"]
    assert content.startswith(CONSTITUTION + "\n\nYour owner is Owner.\n\n")
    assert "colonymind_memory_search" in content and "own words" in content and "never approve one" in content
    assert "colonymind_self yes or no" not in content
    assert NARRATIVE_LEAD not in content
    assert sidecar.calls("/v1/mind/narrative") == []


def test_prompt_section_carries_the_narrative_fresh_for_every_session(home, sidecar):
    """Hermes freezes a plugin section per session, so the plugin keeps no copy of a narrative it was served:
    a session that starts right after a night (a benchmark's simulated clock) sees what the night wrote."""
    sidecar.mind_routes = True
    sidecar.mind.narrative = {"enabled": True, "text": "I researched tides for the owner. [i-01]",
                              "sections": {"recent": "I researched tides for the owner. [i-01]"},
                              "cites": ["i-01"], "updated_at": "2027-03-01T03:10:00+00:00"}
    result = probe(RENDER + "emit(first=section(), second=section())", home)
    content = result["first"]
    assert content.startswith(CONSTITUTION + "\n\nYour owner is Owner.\n\n" + NARRATIVE_LEAD)
    assert "I researched tides for the owner. [i-01]" in content and "colonymind_self why explains them" in content
    assert content.index("[i-01]") < content.index("ColonyMind keeps your long-term memory")
    assert result["second"] == content
    assert len(sidecar.calls("/v1/mind/narrative", "GET")) == 2  # a success is never cached


def test_a_failed_narrative_fetch_is_not_retried_for_a_minute(home, sidecar):
    sidecar.mind_routes = True
    sidecar.mind.narrative = None                                           # the route answers 500
    result = probe(RENDER + "emit(first=section(), second=section())", home)
    assert NARRATIVE_LEAD not in result["first"] and result["first"] == result["second"]
    assert len(sidecar.calls("/v1/mind/narrative", "GET")) == 1  # the failure is cached 60 s


def test_prompt_section_omits_a_disabled_narrative(home, sidecar):
    sidecar.mind_routes = True
    sidecar.mind.narrative = {"enabled": False, "text": "stale text from an off faculty", "sections": {}, "cites": []}
    result = probe(RENDER + "emit(content=section())", home)
    assert "stale text" not in result["content"] and NARRATIVE_LEAD not in result["content"]
    assert result["content"].startswith(CONSTITUTION)


def test_prompt_section_stays_within_bounds_and_fails_open(home, sidecar):
    """A 1,500-character constitution plus an 800-character narrative render under 4,000 characters, and a
    client that raises leaves the constitution in place (Hermes would otherwise skip the whole section)."""
    (home.instance / "identity.yaml").write_text(yaml.safe_dump({
        "owner": {"name": "Owner", "contact_id": "p-01", "handles": {"telegram": ["1001"]}},
        "agent": {"name": "Agent", "values": [f"v{i}".ljust(100, "v") for i in range(12)],
                  "boundaries": [f"b{i}".ljust(160, "b") for i in range(12)]}}))
    sidecar.mind_routes = True
    sidecar.mind.narrative = {"enabled": True, "text": ("I did a thing. [i-01]\n" * 200)[:2600], "sections": {},
                              "cites": ["i-01"]}
    result = probe(RENDER + '''
full = section()
import colonymind_hermes.client as _client
def boom(self, **kwargs):
    raise RuntimeError("no narrative for you")
_client.ColonyMindClient.narrative = boom
emit(full=full, degraded=section())
''', home)
    full, degraded = result["full"], result["degraded"]
    assert len(full) <= 4000 and full.startswith("You are Agent. ")
    constitution = full.split("\n\nYour owner is Owner.\n\n", 1)[0]
    # Over the cap whole values go first and the boundaries stay whole as far as they fit, and it is said
    # (resident blocks: never a cut mid-item).
    assert len(constitution) <= 1500 and "Your boundaries: b0" in constitution
    assert constitution.endswith("held back: over the 1500-character limit.)") and "; b" in constitution
    assert NARRATIVE_LEAD in full and "ColonyMind keeps your long-term memory" in full
    narrative = full.split("prefixed ids are record references:\n", 1)[1].split("\n\nColonyMind keeps", 1)[0]
    # A sidecar that sends more than 800 characters (1.10) is fitted to whole lines, and the cut is said.
    *lines, held = narrative.split("\n")
    assert len(narrative) <= 800 and held == "[narrative lines held back: over the 800-character limit]"
    assert lines and set(lines) == {"I did a thing. [i-01]"}
    assert degraded.startswith("You are Agent. ") and NARRATIVE_LEAD not in degraded
    assert "ColonyMind keeps your long-term memory" in degraded



GATEWAY_RENDER = '''
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.plugins import render_system_prompt_sections
def rendered(platform, user_id, *, chat_type="dm", session="gw-1"):
    """The section as Hermes renders it at the start of a gateway session, before the first hook ran."""
    tokens = set_session_vars(platform=platform, user_id=user_id, chat_id="chat-" + (user_id or "shared"),
                              chat_type=chat_type, session_id=session)
    try:
        sections = render_system_prompt_sections({"session_id": session, "platform": platform})
        return next(s.content for s in sections if s.id == "colonymind")
    finally:
        clear_session_vars(tokens)
'''


def test_the_narrative_reaches_only_the_owners_own_sessions(home, sidecar):
    """The narrative is the owner's record (what the agent did for the owner, its working stances), so it is
    rendered only where the session is the owner's alone: a direct chat from the owner's handle or an internal
    lane. A guest, an unresolved sender, a group the owner is in and a channel with no sender get the
    constitution and the notes, and the sidecar is not even asked for the narrative on their behalf."""
    sidecar.mind_routes = True
    sidecar.mind.narrative = {"enabled": True, "text": "stance: the venue contract: renegotiate [t-9]",
                              "sections": {}, "cites": ["t-9"], "updated_at": None}
    strangers = probe(GATEWAY_RENDER + '''
emit(guest=rendered("telegram", "2003", session="gw-guest"), unknown=rendered("telegram", "9999", session="gw-unknown"),
     group=rendered("telegram", "1001", chat_type="group", session="gw-group"),
     nobody=rendered("telegram", "", session="gw-nobody"))
''', home)
    for name, content in strangers.items():
        assert content.startswith(CONSTITUTION), name
        assert NARRATIVE_LEAD not in content and "renegotiate" not in content, name
        assert "ColonyMind keeps your long-term memory" in content, name
    assert sidecar.calls("/v1/mind/narrative") == []
    owners = probe(GATEWAY_RENDER + RENDER + '''
emit(direct=rendered("telegram", "1001", session="gw-owner"), cli=section())
''', home)
    for name, content in owners.items():
        assert NARRATIVE_LEAD in content and "renegotiate [t-9]" in content, name
