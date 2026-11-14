"""Recall notices use the owner's chosen route through the stock send tool."""

import json

import pytest

from conftest import CONTACTS, build_home, probe as subprocess_probe
from test_provider_seams import PRELUDE, _recalling


OWNER_DM = {"platform": "telegram", "user_id": "1001", "chat_id": "1001", "chat_type": "dm",
            "session_id": "session-1"}
DETAIL_CANARY = "flight to Rome moved to Friday"
DETAILS = [f"• {DETAIL_CANARY} (2026-09-27)", "• wifi password: [withheld] (2026-08-30)",
           "• 1 about another person, not shown"]

# Install the stock module fake before loading the provider. No test can call a real gateway.
NOTICE_PRELUDE = '''
import logging, sys, threading, time, types
SENDS, SEND_THREADS, WARNINGS = [], [], []
SEND_MODE = "success"
SEND_STARTED = threading.Event()
SEND_RELEASE = threading.Event()
SECRET = "PRIVATE-SEND-ERROR-CANARY"
class CaptureWarnings(logging.Handler):
    def emit(self, record):
        if "colonymind" in record.name and record.levelno >= logging.WARNING:
            WARNINGS.append(record.getMessage())
logging.getLogger().addHandler(CaptureWarnings())
def fake_send(args, **kwargs):
    SENDS.append(dict(args))
    SEND_THREADS.append(threading.get_ident())
    SEND_STARTED.set()
    if SEND_MODE == "blocked":
        SEND_RELEASE.wait(60)
    if SEND_MODE == "exception":
        raise RuntimeError(SECRET + str(args))
    if SEND_MODE == "error":
        return json.dumps({"success": False, "error": SECRET + str(args)})
    if SEND_MODE == "malformed":
        return SECRET + str(args)
    if SEND_MODE == "list":
        return json.dumps([SECRET, args])
    if SEND_MODE == "missing_success":
        return json.dumps({"message": SECRET + str(args)})
    return json.dumps({"success": True, "message_id": "notice-1"})
stock_send = types.ModuleType("tools.send_message_tool")
stock_send.send_message_tool = fake_send
sys.modules["tools.send_message_tool"] = stock_send
''' + PRELUDE + '''
from agent.memory_manager import MemoryManager
from gateway.session_context import set_session_vars, clear_session_vars
def initialize(platform="telegram"):
    provider.initialize("session-1", hermes_home=os.environ["HERMES_HOME"], platform=platform)
    manager = MemoryManager()
    manager.add_provider(provider)
    return manager
def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate(), "notice worker did not finish in time"
def status_dict(status):
    return None if status is None else {
        "provider_label": status.provider_label, "count": status.count, "glyph": status.glyph}
'''


@pytest.fixture
def sidecar(tmp_path):
    """The request seam is in memory, so these notices need no local listening socket."""
    class Sidecar:
        assemble = None
        requests = tmp_path / "requests.jsonl"

        def calls(self, path):
            if not self.requests.exists():
                return []
            return [call for line in self.requests.read_text().splitlines()
                    if (call := json.loads(line))["path"] == path]

    return Sidecar()


@pytest.fixture
def home(tmp_path, sidecar):
    profile = build_home(tmp_path, 7777, "http://127.0.0.1:7777")
    profile.notice_sidecar = sidecar
    return profile


def probe(code, home, **kwargs):
    sidecar = home.notice_sidecar
    assembly = sidecar.assemble({})
    transport = f'''
import httpx, importlib
from urllib.parse import urlsplit
provider_module = importlib.import_module(type(provider).__module__)
ASSEMBLY = {assembly!r}
CONTACTS = {CONTACTS!r}
REQUESTS = pathlib.Path({str(sidecar.requests)!r})
def fake_request(timeout, method, url, **kwargs):
    path = urlsplit(url).path
    with REQUESTS.open("a") as stream:
        stream.write(json.dumps({{"path": path, "json": kwargs.get("json")}}) + "\\n")
    if path == "/v1/host/context/assemble":
        payload = ASSEMBLY
    elif path == "/v1/host/contacts/resolve":
        params = kwargs["params"]
        payload = CONTACTS[(params["gateway"], params["address"])]
    elif path == "/v1/host/context/temporal":
        payload = {{"title": "Current Time", "body": "Wednesday at noon"}}
    elif path == "/v1/memory/branches":
        payload = {{"branches": [], "tool_available": False}}
    else:
        raise AssertionError("unexpected sidecar request: " + path)
    return httpx.Response(200, json=payload, request=httpx.Request(method, url))
provider_module._request_within = fake_request
socket.socket.connect = no_network
socket.create_connection = no_network
'''
    return subprocess_probe(transport + code, home, **kwargs)


def _configure(home, destination=None, *, details=False):
    notice = {"details": details}
    if destination is not None:
        notice["destination"] = destination
    home.write_config(plugins={**home.config["plugins"], "colonymind": {
        **home.config["plugins"]["colonymind"], "recall_notice": notice}})


def _assembly(sidecar, *, count=3, details=DETAILS, memory=True):
    _recalling(sidecar)
    original = sidecar.assemble

    def assemble(body):
        result = original(body)
        recalled = result["sections"][1]
        recalled["citations"] = recalled["citations"][:count]
        if not memory:
            result["sections"].pop()
        if details is not None:
            result["recall_notice_details"] = details
        return result

    sidecar.assemble = assemble


@pytest.mark.parametrize("destination,details", [
    (None, False), ("origin", False), ("origin", True), ("off", True),
    ("home", False), ("home", True), ("home", "true"),
    ("telegram:1001:topic-4", False), ("telegram:1001:topic-4", True),
])
def test_notice_destination_and_details_preserve_the_stock_origin_contract(home, sidecar, destination, details):
    _configure(home, destination, details=details)
    _assembly(sidecar)
    external = destination not in (None, "origin", "off")
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
context = manager.prefetch_all("where does the ferry leave from", session_id="session-1")
assert SENDS == [], "prefetch sent before Hermes displayed recall status"
statuses = [status_dict(provider.recall_status()) for _ in range(3)]
indicator = manager.describe_recall()
if {external!r}:
    wait_for(lambda: len(SENDS) == 1)
clear_session_vars(tokens)
emit(context=context, statuses=statuses, indicator=indicator, sends=SENDS, warnings=WARNINGS,
     asynchronous=all(ident != threading.get_ident() for ident in SEND_THREADS))
''', home, prelude=NOTICE_PRELUDE)
    assert "three quotations" in result["context"]
    assert DETAIL_CANARY not in result["context"]
    assert result["warnings"] == []
    if external:
        target = "telegram" if destination == "home" else destination
        message = "🧠 ColonyMind recalled 3 memories (telegram dm)"
        if details is True:
            message += "\n" + "\n".join(DETAILS)
        assert result["sends"] == [{"action": "send", "target": target, "message": message}]
        assert result["asynchronous"] is True
        assert result["statuses"] == [None] * 3
        assert result["indicator"] == ""
    elif destination == "off":
        assert result["statuses"] == [None] * 3
        assert result["indicator"] == ""
        assert result["sends"] == []
    else:
        assert result["statuses"] == [{"provider_label": "ColonyMind", "count": 3, "glyph": "🧠"}] * 3
        assert result["indicator"] == f"🧠 ColonyMind {chr(0x2014)} recalled 3 memories"
        assert result["sends"] == []
    request, = sidecar.calls("/v1/host/context/assemble")
    if external and details is True:
        assert request["json"]["recall_notice_details"] is True
    else:
        assert "recall_notice_details" not in request["json"]


@pytest.mark.parametrize("count,words", [(0, "relevant memory"), (1, "1 memory")])
def test_external_notice_count_uses_the_stock_singular_and_uncounted_wording(home, sidecar, count, words):
    _configure(home, "home")
    _assembly(sidecar, count=count)
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
assert manager.describe_recall() == ""
wait_for(lambda: len(SENDS) == 1)
emit(sends=SENDS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["sends"][0]["message"] == f"🧠 ColonyMind recalled {words} (telegram dm)"


@pytest.mark.parametrize("platform", ["cli", "tui"])
def test_home_notice_on_the_owners_terminal_falls_back_to_origin(home, sidecar, platform):
    _configure(home, "home", details=True)
    _assembly(sidecar)
    result = probe(f'''
manager = initialize({platform!r})
context = manager.prefetch_all("where does the ferry leave from", session_id="session-1")
emit(context=context, indicator=manager.describe_recall(), sends=SENDS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["indicator"] == f"🧠 ColonyMind {chr(0x2014)} recalled 3 memories"
    assert result["sends"] == []
    assert "recall_notice_details" not in sidecar.calls("/v1/host/context/assemble")[0]["json"]


@pytest.mark.parametrize("destination", ["nowhere", "telegram:", ":1001", 14])
def test_unknown_destination_falls_back_to_origin_and_warns_once(home, sidecar, destination):
    _configure(home, destination, details=True)
    _assembly(sidecar)
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
indicators = []
for _ in range(3):
    manager.prefetch_all("where does the ferry leave from", session_id="session-1")
    indicators.extend([manager.describe_recall(), manager.describe_recall()])
emit(indicators=indicators, sends=SENDS, warnings=WARNINGS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["indicators"] == [f"🧠 ColonyMind {chr(0x2014)} recalled 3 memories"] * 6
    assert result["sends"] == []
    assert len(result["warnings"]) == 1
    assert "origin" in result["warnings"][0].lower()
    assert all("recall_notice_details" not in call["json"]
               for call in sidecar.calls("/v1/host/context/assemble"))


@pytest.mark.parametrize("destination", ["home", "telegram:1001"])
def test_notice_and_details_are_only_for_owner_private_turns(home, sidecar, destination):
    _configure(home, destination, details=True)
    _assembly(sidecar)
    turns = [OWNER_DM, {**OWNER_DM, "user_id": "2003", "chat_id": "2003"},
             {**OWNER_DM, "chat_id": "-100500", "chat_type": "group"}, OWNER_DM]
    result = probe(f'''
manager = initialize()
counts, indicators = [], []
for index, binding in enumerate({turns!r}):
    tokens = set_session_vars(**binding)
    before = len(SENDS)
    context = manager.prefetch_all("where does the ferry leave from", session_id="session-1")
    assert len(SENDS) == before
    assert "three quotations" in context
    indicators.append(manager.describe_recall())
    if index in (0, 3):
        wait_for(lambda: len(SENDS) == before + 1)
    counts.append(len(SENDS))
    clear_session_vars(tokens)
emit(counts=counts, indicators=indicators, sends=SENDS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["counts"] == [1, 1, 1, 2]
    assert result["indicators"] == [""] * 4
    assert all(DETAIL_CANARY in send["message"] for send in result["sends"])
    assert [call["json"].get("recall_notice_details")
            for call in sidecar.calls("/v1/host/context/assemble")] == [True, None, None, True]
    assert [call["json"]["context"]["metadata"]["owner_private_turn"]
            for call in sidecar.calls("/v1/host/context/assemble")] == [True, False, False, True]


def test_worker_does_not_request_details_or_send_a_notice(home, sidecar):
    _configure(home, "telegram:1001", details=True)
    _assembly(sidecar)
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
emit(status=status_dict(provider.recall_status()), sends=SENDS)
''', home, env={"HERMES_KANBAN_TASK": "worker-1"}, prelude=NOTICE_PRELUDE)
    assert result == {"status": None, "sends": []}
    assert "recall_notice_details" not in sidecar.calls("/v1/host/context/assemble")[0]["json"]


def test_status_uses_the_turn_route_captured_before_gateway_context_is_cleared(home, sidecar):
    _configure(home, "home")
    _assembly(sidecar)
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
clear_session_vars(tokens)
assert provider.recall_status() is None
wait_for(lambda: len(SENDS) == 1)
emit(sends=SENDS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["sends"] == [{"action": "send", "target": "telegram",
                                 "message": "🧠 ColonyMind recalled 3 memories (telegram dm)"}]


def test_notice_worker_preserves_context_variables_used_by_stock_profile_routing(home, sidecar):
    _configure(home, "home")
    _assembly(sidecar)
    result = probe(f'''
from contextvars import ContextVar
from gateway.session_context import get_session_env
active_profile = ContextVar("active_profile", default="wrong-default-profile")
CONTEXTS = []
def profile_send(args, **kwargs):
    CONTEXTS.append({{"profile": active_profile.get(),
                     "sender": get_session_env("HERMES_SESSION_USER_ID", "")}})
    return fake_send(args, **kwargs)
stock_send.send_message_tool = profile_send
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
active_profile.set("owners-selected-profile")
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
assert manager.describe_recall() == ""
wait_for(lambda: len(SENDS) == 1)
emit(contexts=CONTEXTS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["contexts"] == [{"profile": "owners-selected-profile", "sender": "1001"}]


@pytest.mark.parametrize("details", [None, []])
def test_external_notice_without_sidecar_details_is_header_only(home, sidecar, details):
    _configure(home, "home", details=True)
    _assembly(sidecar, details=details)
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
assert manager.describe_recall() == ""
wait_for(lambda: len(SENDS) == 1)
emit(sends=SENDS)
''', home, prelude=NOTICE_PRELUDE)
    assert result["sends"][0]["message"] == "🧠 ColonyMind recalled 3 memories (telegram dm)"


def test_missing_memory_section_never_sends_a_notice(home, sidecar):
    _configure(home, "home", details=True)
    _assembly(sidecar, memory=False)
    result = probe(f'''
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
emit(status=status_dict(provider.recall_status()), sends=SENDS)
''', home, prelude=NOTICE_PRELUDE)
    assert result == {"status": None, "sends": []}


@pytest.mark.parametrize("mode", ["exception", "error", "malformed", "list", "missing_success"])
def test_send_failures_are_nonfatal_redacted_and_rate_limited(home, sidecar, mode):
    _configure(home, "home", details=True)
    _assembly(sidecar)
    result = probe(f'''
SEND_MODE = {mode!r}
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
for attempt in range(3):
    manager.prefetch_all("where does the ferry leave from", session_id="session-1")
    assert provider.recall_status() is None
    assert provider.recall_status() is None
    wait_for(lambda: len(SENDS) == attempt + 1)
    if attempt == 0:
        wait_for(lambda: len(WARNINGS) >= 1)
    time.sleep(0.02)
emit(sends=SENDS, warnings=WARNINGS)
''', home, prelude=NOTICE_PRELUDE)
    assert len(result["sends"]) == 3
    assert len(result["warnings"]) == 1
    logged = "\n".join(result["warnings"])
    assert "PRIVATE-SEND-ERROR-CANARY" not in logged
    assert DETAIL_CANARY not in logged
    assert "recalled 3 memories" not in logged


def test_slow_sender_does_not_block_or_accumulate_workers(home, sidecar):
    _configure(home, "home", details=True)
    _assembly(sidecar)
    result = probe(f'''
SEND_MODE = "blocked"
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
durations = []
for attempt in range(8):
    if attempt == 4:
        provider = type(provider)()
        manager = initialize()
    manager.prefetch_all("where does the ferry leave from", session_id="session-1")
    started = time.monotonic()
    assert provider.recall_status() is None
    assert provider.recall_status() is None
    durations.append(time.monotonic() - started)
    assert SEND_STARTED.wait(1)
emit(sends=SENDS, durations=durations, warnings=WARNINGS)
''', home, prelude=NOTICE_PRELUDE, timeout=15)
    assert len(result["sends"]) == 1
    assert max(result["durations"]) < 0.25
    assert len(result["warnings"]) == 1
    assert DETAIL_CANARY not in "\n".join(result["warnings"])


def test_slow_sender_warns_on_deadline_without_retrying_or_disclosing_text(home, sidecar):
    _configure(home, "home", details=True)
    _assembly(sidecar)
    result = probe(f'''
provider_module._RECALL_SEND_TIMEOUT_S = 0.05
SEND_MODE = "blocked"
manager = initialize()
tokens = set_session_vars(**{OWNER_DM!r})
manager.prefetch_all("where does the ferry leave from", session_id="session-1")
assert provider.recall_status() is None
wait_for(lambda: len(WARNINGS) == 1)
for _ in range(4):
    assert provider.recall_status() is None
time.sleep(0.06)
emit(sends=SENDS, warnings=WARNINGS)
''', home, prelude=NOTICE_PRELUDE, timeout=15)
    assert len(result["sends"]) == 1
    assert len(result["warnings"]) == 1
    assert "timed out" in result["warnings"][0]
    assert DETAIL_CANARY not in result["warnings"][0]
