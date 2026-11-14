"""A provider built for a sender recalls that sender, even before its first hook."""

import pytest

from conftest import CANARY, OWNER, build_home, probe
from test_prefetch import PROVIDER_PRELUDE


# Real HTTP clients and provider request deadlines, with an in-process sidecar transport.
TRANSPORT = '''
import httpx, json, time
from unittest.mock import patch
requests = []
contacts = {"telegram:2002": "p-02", "telegram:2003": "p-03", "tui:2003": "p-03"}
resolve_status = 200
temporal_delay = 0
def respond(request):
    path = request.url.path
    query = dict(request.url.params)
    body = json.loads(request.content) if request.content else None
    requests.append({"path": path, "query": query, "json": body})
    if path == "/v1/host/contacts/resolve":
        contact = contacts.get(query.get("gateway", "") + ":" + query.get("address", ""))
        return httpx.Response(resolve_status if contact else 404, json={"contact_id": contact})
    if path == "/v1/host/context/temporal":
        time.sleep(temporal_delay)
        return httpx.Response(200, json={"title": "Current Time", "body": "Sunday, 10:00 AM UTC"})
    if path == "/v1/host/context/assemble":
        sections = [{"id": "shared", "title": "Shared", "body": "shared facts", "priority": 50}]
        if body.get("audience") != "viewer":
            sections.append({"id": "private", "title": "Owner notes", "body": CANARY, "priority": 90})
        return httpx.Response(200, json={"sections": sections})
    return httpx.Response(404, json={})
client_type = httpx.Client
async_client_type = httpx.AsyncClient
def client(*args, **kwargs):
    return client_type(*args, **{**kwargs, "transport": httpx.MockTransport(respond)})
def async_client(*args, **kwargs):
    return async_client_type(*args, **{**kwargs, "transport": httpx.MockTransport(respond)})
patch("httpx.Client", client).start()
patch("httpx.AsyncClient", async_client).start()
'''

# Real Hermes construction and turns, stopped before the first model request.
AGENTS = '''
from hermes_cli.plugins import get_plugin_manager
get_plugin_manager().discover_and_load(force=True)
for target, value in (("model_tools.get_tool_definitions", []), ("model_tools.check_toolset_requirements", {})):
    patch(target, return_value=value).start()
patch("agent.process_bootstrap.OpenAI").start()
from run_agent import AIAgent
class Stop(BaseException):
    pass
def build(session, **identity):
    agent = AIAgent(model="test-model", api_mode="chat_completions", provider="custom",
                    base_url="https://model.invalid/v1", api_key="test-only-key", quiet_mode=True, max_iterations=3,
                    skip_context_files=True, skip_background_review=True, session_id=session, **identity)
    def stop(*args, **kwargs):
        raise Stop()
    agent._interruptible_streaming_api_call = agent._interruptible_api_call = stop
    return agent
def mine(agent):
    return agent._memory_manager.providers[0]
def turn(agent, message="what do I owe you?"):
    seen = {}
    manager = agent._memory_manager
    original = manager.prefetch_all
    def prefetch_all(query, **kwargs):
        seen["recall"] = original(query, **kwargs)
        return seen["recall"]
    manager.prefetch_all = prefetch_all
    try:
        agent.run_conversation(message)
    except Stop:
        seen["stopped"] = True
    finally:
        manager.prefetch_all = original
    seen["senders"] = dict(mine(agent)._turn_senders)
    return seen
'''


@pytest.fixture
def recall_probe(tmp_path):
    home = build_home(tmp_path, 0, "http://sidecar.invalid")
    prelude = f"CANARY = {CANARY!r}\n" + TRANSPORT + PROVIDER_PRELUDE + '''
original_emit = emit
def emit(**values):
    original_emit(requests=requests, **values)
'''

    def run(code, *, limit=0):
        home.write_config(**{**home.config, "plugins": {**home.config["plugins"], "hook_callback_timeout": limit}})
        return probe(code, home, prelude=prelude)

    return run


def assembled(result):
    return [(call["json"]["context"]["session_id"], call["json"]["context"]["contact_id"],
             call["json"].get("audience")) for call in result["requests"]
            if call["path"] == "/v1/host/context/assemble"]


@pytest.mark.parametrize("limit", [0, 0.5])
def test_a_named_senders_first_turn_recalls_only_that_contact(recall_probe, limit):
    """A slow earlier callback must not give a contact the owner's recall when Hermes skips its hook."""
    result = recall_probe(AGENTS + '''
import logging
records = []
class Keep(logging.Handler):
    def emit(self, record):
        if "pre_llm_call" in record.getMessage():
            records.append(record.getMessage())
logging.getLogger().addHandler(Keep())
temporal_delay = 2.0
other = build("owner-1", platform="tui")
contact = build("contact-1", platform="tui", user_id="2003")
slow = turn(other, "hi")
first = turn(contact)
emit(slow=slow, first=first, log=records)
''', limit=limit)
    assert result["slow"]["stopped"] and result["first"]["stopped"]
    if limit:
        assert any("timed out after 0.5s" in line for line in result["log"])
        assert any("skipped after previous timeout" in line for line in result["log"])
        assert result["first"]["senders"] == {}
    else:
        assert result["first"]["senders"] == {"contact-1": ["tui", "2003"]}
    assert CANARY not in result["first"]["recall"]
    assert "shared facts" in result["first"]["recall"]
    assert assembled(result) == [("contact-1", "p-03", "viewer")]


@pytest.mark.parametrize("platform", ["tui", "telegram", None])
def test_initial_sender_resolves_the_same_contact_as_the_agents_hook(recall_probe, platform):
    """Both paths pair the sender with Hermes' effective platform, including the CLI default."""
    effective = platform or "cli"
    result = recall_probe(AGENTS + f'''
contacts[{effective + ":2003"!r}] = {"contact-" + effective!r}
agent = build("contact-1", platform={platform!r}, user_id="2003")
before = mine(agent).prefetch("what do I owe you?", session_id=agent.session_id)
hooked = turn(agent)
emit(before=before, hooked=hooked, context=mine(agent)._turn_sender_context())
''')
    assert CANARY not in result["before"] and "shared facts" in result["before"]
    assert result["hooked"]["stopped"] and "shared facts" in result["hooked"]["recall"]
    assert CANARY not in result["hooked"]["recall"]
    assert result["hooked"]["senders"] == {"contact-1": [platform or "", "2003"]}
    assert result["context"] == [effective, "2003", ""]
    assert assembled(result) == [("contact-1", "contact-" + effective, "viewer")] * 2
    assert all(call["json"]["context"]["channel_id"] == effective for call in result["requests"]
               if call["path"] == "/v1/host/context/assemble")
    lookup = next(call for call in result["requests"] if call["path"] == "/v1/host/contacts/resolve")
    assert lookup["query"]["gateway"] == effective and lookup["query"]["address"] == "2003"


@pytest.mark.parametrize("status", [404, 503])
def test_an_unresolved_initial_sender_recalls_nothing(recall_probe, status):
    result = recall_probe(f'''
resolve_status = {status}
provider.initialize("contact-1", platform="tui", user_id="2003")
emit(context=provider.prefetch("what do I owe you?", session_id="contact-1"))
''')
    assert result["context"] == ""
    assert any(call["path"] == "/v1/host/contacts/resolve" for call in result["requests"])
    assert assembled(result) == []


@pytest.mark.parametrize("gateway_sender, expected", [("2002", "p-02"), ("", None)])
def test_gateway_context_keeps_precedence(recall_probe, gateway_sender, expected):
    result = recall_probe(f'''
provider.initialize("session-1", platform="tui", user_id="2003")
provider.resolve_contact("telegram", "1001", "session-1", warm=False)
tokens = set_session_vars(platform="telegram", user_id={gateway_sender!r}, chat_id="group-1", session_id="session-1")
context = provider.prefetch("what do I owe you?", session_id="session-1")
clear_session_vars(tokens)
emit(context=context)
''')
    assert CANARY not in result["context"]
    if expected:
        assert "shared facts" in result["context"]
        assert assembled(result) == [("session-1", expected, "viewer")]
    else:
        assert result["context"] == "" and assembled(result) == []


def test_hook_sender_keeps_precedence(recall_probe):
    result = recall_probe(AGENTS + '''
from agent.subagent_lifecycle import bind_subagent_parent
from hermes_cli.lifecycle import invoke_hook
agent = build("contact-1", platform="tui", user_id="2003")
with bind_subagent_parent(agent):
    invoke_hook("pre_llm_call", session_id=agent.session_id, task_id="t", turn_id="turn",
                user_message="what do I owe you?", conversation_history=[], is_first_turn=True,
                model="m", platform="telegram", parent_session_id="", sender_id="2002")
context = mine(agent).prefetch("what do I owe you?", session_id=agent.session_id)
emit(context=context, senders=mine(agent)._turn_senders)
''')
    assert result["senders"] == {"contact-1": ["telegram", "2002"]}
    assert CANARY not in result["context"] and "shared facts" in result["context"]
    assert assembled(result) == [("contact-1", "p-02", "viewer")]


@pytest.mark.parametrize("platform", ["", "cli", "tui", "internal", "system", "owner", "api", "worker", "cron"])
def test_internal_platform_without_a_sender_keeps_owner_recall(recall_probe, platform):
    result = recall_probe(f'''
provider.initialize("session-1", platform={platform!r})
emit(context=provider.prefetch("what are my plans", session_id="session-1"))
''')
    assert CANARY in result["context"]
    assert assembled(result) == [("session-1", OWNER, None)]


@pytest.mark.parametrize("platform", ["telegram", "discord", "desktop", "api_server", "subagent"])
def test_real_channel_without_a_sender_keeps_recall_empty(recall_probe, platform):
    result = recall_probe(f'''
provider.initialize("session-1", platform={platform!r})
emit(context=provider.prefetch("what are my plans", session_id="session-1"))
''')
    assert result["context"] == "" and assembled(result) == []
