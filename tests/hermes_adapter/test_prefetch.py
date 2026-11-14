"""Memory provider: prefetch works, guests get the scoped projection (evals 7.2 test 8 canary)."""

from conftest import API_KEY, CANARY, OWNER, probe

PROVIDER_PRELUDE = '''
import json, os
def emit(**values):
    print("@@RESULT@@" + json.dumps(values, default=str), flush=True)
from plugins.memory import load_memory_provider
provider = load_memory_provider("colonymind-memory")
assert provider is not None, "provider did not load"
provider.initialize("session-1", hermes_home=os.environ["HERMES_HOME"], platform="cli")
from gateway.session_context import set_session_vars, clear_session_vars
'''


def test_provider_loads_from_shared_keys_and_prefetches_for_the_owner(home, sidecar):
    result = probe('''
context = provider.prefetch("what are my plans", session_id="session-1")
emit(name=provider.name, url=provider.sidecar_url, key=provider._api_key, context=context,
     tools=[s["name"] for s in provider.get_tool_schemas()])
''', home, prelude=PROVIDER_PRELUDE)
    assert result["name"] == "colonymind"
    assert result["url"] == sidecar.url and result["key"] == API_KEY
    assert CANARY in result["context"] and "shared facts" in result["context"]
    assert "## Current Time" in result["context"]
    assert "colonymind_claim_task" not in result["tools"] and "colonymind_resolve_commitment" in result["tools"]
    assert "Persistent state" not in result["context"] and "[priority" not in result["context"]  # no per-turn preamble
    call, = sidecar.calls("/v1/host/context/assemble", "POST")
    assert call["authorization"] == f"Bearer {API_KEY}"
    assert call["json"]["context"]["contact_id"] == OWNER and "audience" not in call["json"]
    assert call["json"]["context"]["metadata"] == {"owner_private_turn": True}


def test_prefetch_names_the_hosts_inline_allowance(home, sidecar):
    """Hermes spills a prefetch longer than ``hooks.output_spill.max_chars`` (10,000 by default) to a file and shows
    a preview. The provider tells the sidecar how much it can put inline, less the room its own Current Time block
    takes, so the sidecar fits the whole context, owner-critical items and memory included, within it."""
    run = 'provider.prefetch("what are my plans", session_id="session-1")\nemit(ok=True)'
    probe(run, home, prelude=PROVIDER_PRELUDE)
    default = sidecar.calls("/v1/host/context/assemble", "POST")[-1]["json"]["inline_chars"]
    home.write_config(**{**home.config, "hooks": {"output_spill": {"max_chars": 65536}}})
    probe(run, home, prelude=PROVIDER_PRELUDE)
    raised = sidecar.calls("/v1/host/context/assemble", "POST")[-1]["json"]["inline_chars"]
    home.write_config(**{**home.config, "hooks": {"output_spill": {"enabled": False}}})
    probe(run, home, prelude=PROVIDER_PRELUDE)
    unbounded = sidecar.calls("/v1/host/context/assemble", "POST")[-1]["json"]
    assert 9_000 <= default < 10_000 and 64_536 <= raised < 65_536 and raised - default == 55_536
    assert "inline_chars" not in unbounded


def test_prefetch_names_how_long_the_host_waits(home, sidecar):
    """Hermes waits a bounded time for an external prefetch and drops a later answer. The provider tells the
    sidecar what is left of that wait, less its own margin, and the sidecar's recall spends only that (its turn
    budget)."""
    result = probe('''
from agent.memory_manager import _EXTERNAL_PREFETCH_TIMEOUT_S as wait
provider.prefetch("what are my plans", session_id="session-1")
emit(wait=wait)
''', home, prelude=PROVIDER_PRELUDE)
    sent = sidecar.calls("/v1/host/context/assemble", "POST")[-1]["json"]["wait_ms"]
    assert (result["wait"] - 0.5) * 1000 - 500 < sent <= (result["wait"] - 0.5) * 1000


def test_the_whole_prefetch_ends_before_the_host_stops_waiting(home, sidecar):
    """Hermes waits a bounded time for the whole prefetch, not only for the assembly: resolving the turn's
    participant and refreshing the contact's clock count too, and a later answer is dropped. With a slow
    participant lookup, a sidecar that spends every millisecond it is offered and a clock refresh that hangs, the
    context still reaches Hermes in time (its clock then from the provider's own time). Driven through Hermes' own
    memory manager, with a shorter wait so the test stays quick."""
    sidecar.delays["/v1/host/contacts/resolve"] = 0.6
    sidecar.delays["/v1/host/context/assemble"] = lambda request: request["json"].get("wait_ms", 10_000) / 1000 - 0.05
    sidecar.delays["/v1/host/context/temporal"] = 6.0
    result = probe('''
import time
import agent.memory_manager as memory_manager
memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S = 3.0
manager = memory_manager.MemoryManager(external_prefetch_timeout=3.0)
manager.add_provider(provider)
set_session_vars(platform="telegram", user_id="1001", chat_id="1001", session_id="session-1")
started = time.monotonic()
context = manager.prefetch_all("what are my plans", session_id="session-1")
emit(context=context, elapsed=time.monotonic() - started)
''', home, prelude=PROVIDER_PRELUDE)
    assert CANARY in result["context"] and "## Current Time" in result["context"], result
    assert result["elapsed"] < 3.0
    assert sidecar.calls("/v1/host/contacts/resolve") and sidecar.calls("/v1/host/context/assemble", "POST")


def test_answers_in_slow_chunks_still_end_by_the_hosts_wait(home, sidecar):
    """An HTTP client's timeout bounds each read, not the request, so an answer that trickles in chunk by chunk
    could take the whole allowance again for every chunk. Here the participant lookup, the assembly and the clock
    refresh each send their body in small chunks over 1.2 s: the lookup and the assembly fit Hermes' wait, the
    clock refresh does not and is cut off at the prefetch's deadline, and the context reaches Hermes in time."""
    for path in ("/v1/host/contacts/resolve", "/v1/host/context/assemble", "/v1/host/context/temporal"):
        sidecar.drips[path] = 1.2
    result = probe('''
import time
import agent.memory_manager as memory_manager
memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S = 3.0
manager = memory_manager.MemoryManager(external_prefetch_timeout=3.0)
manager.add_provider(provider)
set_session_vars(platform="telegram", user_id="1001", chat_id="1001", session_id="session-1")
started = time.monotonic()
context = manager.prefetch_all("what are my plans", session_id="session-1")
emit(context=context, elapsed=time.monotonic() - started)
''', home, prelude=PROVIDER_PRELUDE)
    assert CANARY in result["context"] and "Agent's time:" in result["context"], result
    assert "Sunday, 10:00 AM UTC" not in result["context"] and result["elapsed"] < 3.0
    assert sidecar.calls("/v1/host/context/temporal")


def test_a_section_the_last_turn_carried_is_one_line_through_stock_hermes(home, sidecar):
    """Stock Hermes' memory manager: the first turn gets every section; once that turn completed
    (``sync_all``, same message), the next turn of the session gets an unchanged section as one line and
    a changed one whole; a new session starts from nothing again."""
    result = probe('''
from agent.memory_manager import MemoryManager
manager = MemoryManager()
manager.add_provider(provider)
first = manager.prefetch_all("what are my plans", session_id="session-1")
manager.sync_all("what are my plans", "Here they are.", session_id="session-1")
assert manager.flush_pending(timeout=10)
second = manager.prefetch_all("and after that?", session_id="session-1")
manager.on_session_switch("session-2", parent_session_id="session-1", reset=True)
third = manager.prefetch_all("and after that?", session_id="session-2")
emit(first=first, second=second, third=third)
''', home, prelude=PROVIDER_PRELUDE)
    assert CANARY in result["first"] and "unchanged since your last turn" not in result["first"]
    assert "unchanged since your last turn: Owner notes" in result["second"] and CANARY not in result["second"]
    assert "unchanged since your last turn: Shared" in result["second"] and "## Current Time" in result["second"]
    assert CANARY in result["third"] and "unchanged since your last turn" not in result["third"]


def test_a_kanban_workers_prefetch_names_its_task(home, sidecar):
    """A worker's prefetch reaches the sidecar on the owner's lane with its task body as the message. It names
    its kanban task, so the sidecar treats it as task work, not the owner's turn: the body carries its own
    lessons, and no owner verdict can ever score a turn lesson in a worker session."""
    probe('''
provider.prefetch("Research order codes for order 6633.", session_id="worker-run-1")
emit(ok=True)
''', home, prelude=PROVIDER_PRELUDE, env={"HERMES_KANBAN_TASK": "t_abc", "HERMES_KANBAN_WORKSPACE": str(home.root)})
    call, = sidecar.calls("/v1/host/context/assemble", "POST")
    assert call["json"]["context"]["contact_id"] == OWNER
    assert call["json"]["context"]["metadata"] == {"kanban_task": "t_abc", "owner_private_turn": False}


def test_a_scheduled_runs_prefetch_says_so(home, sidecar):
    """Stock Hermes builds a cron job's agent with ``platform="cron"`` (``cron.scheduler``) and runs it on the
    owner's lane with the job's prompt as the message. Its prefetch says it is a scheduled run, so the sidecar
    never treats it as the owner's own turn."""
    probe('''
provider.initialize("cron_job_1", hermes_home=os.environ["HERMES_HOME"], platform="cron")
provider.prefetch("Summarise the new items.", session_id="cron_job_1")
emit(ok=True)
''', home, prelude=PROVIDER_PRELUDE)
    call, = sidecar.calls("/v1/host/context/assemble", "POST")
    assert call["json"]["context"]["contact_id"] == OWNER
    assert call["json"]["context"]["metadata"] == {"run": "cron", "owner_private_turn": False}


def test_guest_prefetch_carries_the_viewer_scope_and_no_owner_only_text(home, sidecar):
    result = probe('''
tokens = set_session_vars(platform="telegram", user_id="2003", chat_id="2003", session_id="session-1")
context = provider.prefetch("what are the plans", session_id="session-1")
clear_session_vars(tokens)
emit(context=context)
''', home, prelude=PROVIDER_PRELUDE)
    assert CANARY not in result["context"]
    assert "shared facts" in result["context"]
    call, = sidecar.calls("/v1/host/context/assemble", "POST")
    assert call["json"]["context"]["contact_id"] == "p-03"
    assert call["json"]["audience"] == "viewer"
    assert "projection_policy" not in call["json"]


def test_unresolved_channel_sender_gets_no_context(home, sidecar):
    result = probe('''
provider._resolve_handle = lambda platform, sender: None
tokens = set_session_vars(platform="telegram", user_id="7777", chat_id="7777", session_id="session-1")
context = provider.prefetch("hello", session_id="session-1")
clear_session_vars(tokens)
emit(context=context)
''', home, prelude=PROVIDER_PRELUDE)
    assert result["context"] == ""
    assert sidecar.calls("/v1/host/context/assemble") == []


def test_provider_never_writes_turns_when_the_general_plugin_is_absent(home, sidecar):
    home.write_config(**{**home.config, "plugins": {**home.config["plugins"], "enabled": []}})
    result = probe('''
provider.sync_turn("remember X", "noted", session_id="session-1", turn_id="turn-1")
provider.shutdown()
emit(diagnostics=provider.get_diagnostics())
''', home, prelude=PROVIDER_PRELUDE)
    assert result["diagnostics"]["turn_writer"] == "read-only"
    assert sidecar.calls("/v1/host/turns/sync", "POST") == []


def test_provider_only_cli_capture_is_absent_regardless_of_platform_grants(home, sidecar):
    """The general plugin owns capture; grants cannot enable a provider writer."""
    home.write_config(**{**home.config, "plugins": {**home.config["plugins"], "enabled": []}})
    run = '''
provider.sync_turn("remember X", "noted", session_id="session-1", turn_id="turn-1")
provider.shutdown()
emit(diagnostics=provider.get_diagnostics())
'''
    for platforms in ({"cli"}, {"whatsapp", "sms", "rcs", "voice"}):
        sidecar.owner_platforms = platforms
        diagnostics = probe(run, home, prelude=PROVIDER_PRELUDE)["diagnostics"]
        assert sidecar.calls("/v1/host/turns/sync", "POST") == []
        assert sidecar.told == {}
        assert diagnostics["turn_writer"] == "read-only" and diagnostics["attribution_refusals"] == 0


def test_guest_direct_tools_are_withheld(home, sidecar):
    """A guest turn is offered no direct tool; a call that still arrives is refused once, terminally."""
    result = probe('''
tokens = set_session_vars(platform="telegram", user_id="2003", chat_id="2003", session_id="session-1")
guest = json.loads(provider.handle_tool_call("colonymind_resolve_commitment", {"commitment_id": "c-01", "action": "fulfilled"}))
guest_tools = [s["name"] for s in provider.get_tool_schemas()]
clear_session_vars(tokens)
tokens = set_session_vars(platform="telegram", user_id="1001", chat_id="1001", session_id="session-2")
owner_tools = [s["name"] for s in provider.get_tool_schemas()]
clear_session_vars(tokens)
emit(guest=guest, guest_tools=guest_tools, owner_tools=owner_tools)
''', home, prelude=PROVIDER_PRELUDE)
    assert result["guest"] == {"unavailable": True, "retry": False, "reason": result["guest"]["reason"]}
    assert "owner-only" in result["guest"]["reason"]
    assert result["guest_tools"] == []
    assert set(result["owner_tools"]) == {"colonymind_resolve_commitment"}
    assert not any(call["path"].startswith("/v1/host/commitments") for call in sidecar.requests)


def test_unbound_channel_session_is_built_without_direct_tools(home, sidecar):
    """A provider initialised for a real channel with no sender (an inbound the gateway could not
    bind) hands Hermes no direct tool at agent build time, so none can burn the turn's iterations."""
    result = probe('''
from plugins.memory import load_memory_provider
unbound = load_memory_provider("colonymind-memory")
unbound.initialize("session-9", hermes_home=os.environ["HERMES_HOME"], platform="capture")
emit(tools=[s["name"] for s in unbound.get_tool_schemas()],
     call=json.loads(unbound.handle_tool_call("colonymind_resolve_commitment", {"commitment_id": "c", "action": "fulfilled"})),
     cli_tools=[s["name"] for s in provider.get_tool_schemas()])
''', home, prelude=PROVIDER_PRELUDE)
    assert result["tools"] == []
    assert result["call"]["unavailable"] is True and result["call"]["retry"] is False
    assert "colonymind_resolve_commitment" in result["cli_tools"]
    assert "colonymind_list_goals" not in result["cli_tools"] and "colonymind_get_patterns" not in result["cli_tools"]


def test_a_fact_told_on_one_channel_reaches_the_owners_prefetch_on_another(home, sidecar):
    """Build plan M8 acceptance 1 through the adapter: the owner's senders on two platforms resolve to one
    contact, so what they said in session A on one channel is recalled in session B on the other (the real
    recall behind it is test_memory_identity_acceptance); a guest's prefetch never gets it."""
    sidecar.contacts[("signal", "+15550001")] = {"contact_id": OWNER, "display_name": "Owner",
                                                "interaction_allowed": True, "trust_tier": "GENESIS"}
    # The general plugin has already captured this fact in the source store.
    sidecar.told[OWNER] = ["My office is room 4, by the lifts."]
    result = probe('''
tokens = set_session_vars(platform="signal", user_id="+15550001", chat_id="+15550001", session_id="sg-b")
owner = provider.prefetch("Which room is my office in?", session_id="sg-b")
clear_session_vars(tokens)
tokens = set_session_vars(platform="telegram", user_id="2003", chat_id="2003", session_id="guest-c")
guest = provider.prefetch("Which room is the office in?", session_id="guest-c")
clear_session_vars(tokens)
emit(owner=owner, guest=guest)
''', home, prelude=PROVIDER_PRELUDE)
    assert "room 4, by the lifts" in result["owner"] and "room 4" not in result["guest"]
    assert sidecar.calls("/v1/host/turns/sync", "POST") == []
    assembled = sidecar.calls("/v1/host/context/assemble", "POST")
    assert [(c["json"]["context"]["contact_id"], c["json"]["context"]["session_id"]) for c in assembled] == [
        (OWNER, "sg-b"), ("p-03", "guest-c")]


def test_a_sender_bound_on_the_agent_alone_reaches_recall(home, sidecar):
    """A host that binds the sender on the agent only (``AIAgent(user_id=...)``, no gateway session context)
    initializes the provider for that sender, so a contact's turn is recalled as that contact even without
    a hook naming the sender. Where the gateway binds a turn, its own sender wins, and a channel with no
    sender stays unbound whatever the agent was built with."""
    prelude = PROVIDER_PRELUDE.replace(
        'provider.initialize("session-1", hermes_home=os.environ["HERMES_HOME"], platform="cli")', '''
from types import SimpleNamespace
from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _memory_provider_init_kwargs
agent = SimpleNamespace(session_id="session-1", _session_db=None, session_cwd=None,
                        **{"_" + key: None for key in _GATEWAY_IDENTITY_PARAMS})
agent._user_id = "2003"
provider.initialize(**_memory_provider_init_kwargs(agent, "telegram"))
''')
    result = probe('''
from hermes_cli.lifecycle import invoke_hook
def hook(session, platform="telegram"):
    return invoke_hook("pre_llm_call", session_id=session, task_id="t", turn_id="turn", user_message="what do I owe you?",
                       conversation_history=[], is_first_turn=True, model="m", platform=platform,
                       parent_session_id="")
clock = hook("session-1")
agent_only = provider.prefetch("what do I owe you?", session_id="session-1")
hook("session-1")
tokens = set_session_vars(platform="telegram", user_id="2002", chat_id="2002", session_id="session-1")
gateway = provider.prefetch("what do I owe you?", session_id="session-1")
clear_session_vars(tokens)
tokens = set_session_vars(platform="telegram", user_id="", chat_id="group-1", session_id="session-1")
channel = provider.prefetch("what do I owe you?", session_id="session-1")
clear_session_vars(tokens)
emit(clock=clock, agent_only=agent_only, gateway=gateway, channel=channel)
''', home, prelude=prelude)
    assert "shared facts" in result["agent_only"] and CANARY not in result["agent_only"]
    assert not any("Current Time for this turn" in str(item) for item in result["clock"])  # recall carries it
    assert result["channel"] == ""
    contacts = [call["json"]["context"]["contact_id"] for call in sidecar.calls("/v1/host/context/assemble", "POST")]
    assert contacts == ["p-03", "p-02"]
    assert all(call["json"]["audience"] == "viewer" for call in sidecar.calls("/v1/host/context/assemble", "POST"))
