"""The host owner's turn on Hermes 0.22: one decision, taken alike by the memory provider and the capture plugin.

Hermes 0.22 runs the terminal, the TUI, the desktop app and a dashboard login as local direct sessions of the
operating-system user and names that user ``uid:<n>`` in the gateway's identity, in the agent it builds and in the
``pre_llm_call`` hook. It stages task workers, forwarded agent-to-agent turns, hosted rooms and editor clients the
same way under other labels, and it lets bots place messages in the owner's own Bot Chat, naming their author on
the agent before the hooks run. Hermes 0.21 names no such user: its terminal hook carries no sender at all.
"""
import importlib
import importlib.util
import itertools
from pathlib import Path
import sys
import types

import pytest

from colonymind_shared import host_owner
from test_colonymind_memory_provider import (_FakeHttpx, _host_owner, _install_session_context,
                                             _load_provider_module, _make_provider)

SUBJECT, OWNER = "uid:1000", "cid-base"
LOCAL = ("local", SUBJECT, "local-session", "dm")
NO_GATEWAY = ("", "", "", "")
STARTED = (SUBJECT, "local-session", "dm")
OWNER_AUTHOR = {"id": SUBJECT, "name": "runner", "is_bot": False}
BOT_AUTHOR = {"id": "bot:helper", "name": "Helper", "is_bot": True}
PEER_AUTHOR = {"id": "bot:other-host/helper", "name": "Helper", "is_bot": True}
WORKER_AUTHOR = {"id": "kanban:dispatcher", "name": "Kanban dispatcher", "is_bot": True}
OTHER_AUTHOR = {"id": "uid:1001", "name": "someone", "is_bot": False}


@pytest.fixture(scope="module")
def provider_mod():
    return _load_provider_module()


@pytest.fixture
def hermes(monkeypatch):
    """What Hermes 0.22 hands a turn: the gateway's identity, the owner's subject and the agent it runs."""
    set_turn = _install_session_context(monkeypatch)
    _host_owner(monkeypatch, SUBJECT)
    running = {}
    lifecycle = types.ModuleType("agent.subagent_lifecycle")
    lifecycle.get_active_subagent_parent = lambda: running.get("agent")
    if importlib.util.find_spec("agent") is None:
        monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
        # Without Hermes installed nothing answers the provider's question whether a prompt is trivial.
        predicate = types.ModuleType("agent.memory_provider")
        predicate.is_trivial_prompt = lambda text: False
        monkeypatch.setitem(sys.modules, "agent.memory_provider", predicate)
    monkeypatch.setitem(sys.modules, "agent.subagent_lifecycle", lifecycle)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    def turn(*, label="cli", session="s1", bound=LOCAL, started=STARTED, author=OWNER_AUTHOR):
        set_turn(platform=bound[0], sender=bound[1], chat=bound[2], chat_type=bound[3])
        running["agent"] = types.SimpleNamespace(session_id=session, platform=label, _user_id=started[0],
                                                 _chat_id=started[1], _chat_type=started[2], _turn_author=author)
    return turn


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    name = "host_owner_turn_adapter"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins/hermes-plugin")]
    monkeypatch.setitem(sys.modules, name, package)
    module = importlib.import_module(name + ".capture")
    settings = importlib.import_module(name + ".client").Settings(
        "http://unreachable.invalid", tmp_path / "key", "test-key", tmp_path, tmp_path, tmp_path / "outbox.sqlite3")
    monkeypatch.setattr(settings, "owner_contact_id", lambda: OWNER)

    class Client:
        def __init__(self):
            self.bodies, self.resolved = [], []

        def resolve_contact(self, platform, sender, **options):
            self.resolved.append((platform, sender, bool(options.get("create"))))
            return {"contact_id": "cid-shadow"} if options.get("create") else None

        def post(self, path, *, json, **_):
            self.bodies.append((path, json))
            return types.SimpleNamespace(is_success=True, status_code=200, json=lambda: {"accepted": True, "work_recorded": True})

    client = Client()
    sessions = module.SessionMap(settings, client)
    capture = module.Capture(sessions, module.TurnOutbox(settings.outbox_path), settings)

    def run_turn(label, *, session="s1", turn="t1", sender=SUBJECT, words="Send me the report by 3pm."):
        values = {"session_id": session, "turn_id": turn, "platform": label, "user_message": words}
        capture.pre_llm_call(**values, sender_id=sender)
        capture.post_llm_call(**values, assistant_response="I will have it ready.")
        module.TurnOutbox(settings.outbox_path).drain(
            lambda payload: module.deliver(client, sessions, settings, payload))

    yield types.SimpleNamespace(module=module, client=client, sessions=sessions, capture=capture, run_turn=run_turn)
    for key in list(sys.modules):
        if key.startswith(name + "."):
            del sys.modules[key]


def turns(client):
    return [body for path, body in client.bodies if path == "/v1/host/turns/sync"]


# -- the decision ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facts,owner", [
    ({}, True),
    ({"label": "tui"}, True),
    ({"label": "desktop"}, True),
    ({"label": "a2a"}, False),
    ({"label": "bot_room"}, False),
    ({"label": "acp"}, False),
    ({"label": "cron"}, False),
    ({"label": "kanban"}, False),
    ({"label": ["cli"]}, False),
    ({"bound": ("telegram", SUBJECT, "local-session", "dm")}, False),
    ({"bound": ("local", "uid:1001", "local-session", "dm")}, False),
    ({"bound": ("local", SUBJECT, "local-session", "group")}, False),
    ({"bound": ("local", SUBJECT, "", "dm")}, False),
    ({"bound": NO_GATEWAY}, True),
    ({"bound": NO_GATEWAY, "label": "tui"}, False),
    ({"bound": NO_GATEWAY, "hook": ("cli", SUBJECT)}, True),
    ({"bound": NO_GATEWAY, "hook": ("cli", "uid:1001")}, False),
    ({"bound": NO_GATEWAY, "started": ("uid:1001", "local-session", "dm")}, False),
    ({"bound": NO_GATEWAY, "started": (SUBJECT, "local-session", "")}, False),
    ({"author": (SUBJECT, False)}, True),
    ({"author": ("bot:helper", True)}, False),
    ({"author": (SUBJECT, True)}, False),
    ({"author": ("uid:1001", False)}, False),
    ({"author": ("", False)}, False),
])
def test_decision(hermes, facts, owner):
    given = {"label": "cli", "started": STARTED, "bound": LOCAL, **facts}
    assert host_owner.decide(**given) is owner


@pytest.mark.parametrize("absent", ["api", "unreadable", "empty", "not-text", "task"])
def test_decision_fails_closed(hermes, monkeypatch, absent):
    owner = sys.modules["gateway.owner_identity"]
    if absent == "api":
        monkeypatch.setitem(sys.modules, "gateway.owner_identity", None)   # Hermes 0.21
    elif absent == "unreadable":
        monkeypatch.setattr(owner, "host_owner_subject", lambda: 1 / 0)
    elif absent == "empty":
        monkeypatch.setattr(owner, "host_owner_subject", lambda: "")
    elif absent == "not-text":
        monkeypatch.setattr(owner, "host_owner_subject", lambda: 1000)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    assert host_owner.decide(label="cli", started=STARTED, bound=LOCAL) is False
    assert host_owner.decide(label="cli", started=STARTED, bound=NO_GATEWAY, hook=("cli", SUBJECT)) is False


def test_named_author():
    assert host_owner.named_author(None) is None
    assert host_owner.named_author({"id": None, "name": None, "is_bot": False}) is None
    assert host_owner.named_author(OWNER_AUTHOR) == (SUBJECT, False)
    assert host_owner.named_author(BOT_AUTHOR) == ("bot:helper", True)
    assert host_owner.named_author({"id": None, "name": "someone", "is_bot": False}) == ("", False)
    assert host_owner.named_author({"id": 7, "name": None, "is_bot": False}) == ("", False)
    assert host_owner.named_author("bot:helper") is None


def test_running_turn_needs_the_agent_of_that_session(hermes):
    assert host_owner.running_turn() is False            # no agent is running a turn
    hermes()
    assert host_owner.running_turn() is True
    assert host_owner.running_turn("s1") is True
    assert host_owner.running_turn("another-session") is False


# -- the capture plugin, in the shapes Hermes 0.22.7 sends --------------------------------------------------------

@pytest.mark.parametrize("label", ["cli", "tui", "desktop"])
def test_capture_files_the_owners_own_turn_under_the_owner_with_no_sender(hermes, adapter, label):
    hermes(label=label)
    assert adapter.module.session_env() == NO_GATEWAY     # the owner's own lane, as Hermes 0.21 runs the terminal
    adapter.run_turn(label)
    assert adapter.sessions.is_owner("s1") is True
    assert adapter.sessions.contact_id("s1") == OWNER
    assert adapter.sessions.standing("s1") == ("owner", label)
    assert adapter.sessions.owner_only({"session_id": "s1", "platform": "local"}) is True
    (body,) = turns(adapter.client)
    assert body["context"]["contact_id"] == OWNER and body["context"]["channel_id"] == label
    assert "sender" not in body
    assert body["user_message"]["content"] == "Send me the report by 3pm."
    assert adapter.client.resolved == []                  # the subject is never looked up, let alone provisioned


NOT_THE_OWNER = {
    "task worker": {"label": "kanban", "bound": NO_GATEWAY, "author": WORKER_AUTHOR, "task": "t_1"},
    "forwarded agent-to-agent turn": {"label": "a2a", "author": PEER_AUTHOR},
    "hosted room": {"label": "bot_room", "author": BOT_AUTHOR},
    "editor client": {"label": "acp", "author": OWNER_AUTHOR},
    "bot message in the Bot Chat": {"label": "cli", "author": BOT_AUTHOR},
    "peer message in the Bot Chat": {"label": "cli", "author": PEER_AUTHOR},
    "another user's words": {"label": "cli", "author": OTHER_AUTHOR},
}


@pytest.mark.parametrize("shape", NOT_THE_OWNER)
def test_capture_never_makes_a_contact_of_the_host_owners_subject(hermes, adapter, monkeypatch, shape):
    given = dict(NOT_THE_OWNER[shape])
    if given.pop("task", None):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    hermes(**given)
    adapter.run_turn(given["label"])
    assert adapter.sessions.is_owner("s1") is not True
    assert adapter.sessions.contact_id("s1") is None
    assert adapter.sessions.standing("s1")[0] != "owner"
    assert adapter.sessions.owner_only({"session_id": "s1", "platform": "local"}) is False
    assert not turns(adapter.client)                      # nothing is filed under the owner or under a stranger
    assert not [call for call in adapter.client.resolved if call[2]]


def test_capture_follows_each_turn_of_the_bot_chat(hermes, adapter):
    hermes(author=BOT_AUTHOR)
    adapter.run_turn("cli", turn="t1", words="Report from the helper bot.")
    assert adapter.sessions.is_owner("s1") is False and not turns(adapter.client)
    hermes(author=OWNER_AUTHOR)
    adapter.run_turn("cli", turn="t2", words="Thanks, file it under invoices.")
    assert adapter.sessions.is_owner("s1") is True
    assert adapter.sessions.standing("s1") == ("owner", "cli")
    (body,) = turns(adapter.client)
    assert body["context"]["turn_id"] == "t2" and "sender" not in body
    hermes(author=PEER_AUTHOR)
    adapter.run_turn("cli", turn="t3", words="Another delivery.")
    assert adapter.sessions.is_owner("s1") is False
    assert adapter.sessions.standing("s1")[0] != "owner"
    assert len(turns(adapter.client)) == 1
    assert not [call for call in adapter.client.resolved if call[2]]


def test_capture_on_hermes_021_is_unchanged(hermes, adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "gateway.owner_identity", None)
    hermes(label="cli", bound=NO_GATEWAY, started=("", "", ""), author=None)
    adapter.run_turn("cli", sender="")                    # the terminal: no sender, the owner's own lane
    assert adapter.sessions.is_owner("s1") is True
    (body,) = turns(adapter.client)
    assert body["context"]["contact_id"] == OWNER and "sender" not in body
    hermes(label="telegram", session="s2", bound=("telegram", "4242", "4242", "dm"), started=("4242", "4242", "dm"))
    assert adapter.module.session_env() == ("telegram", "4242", "4242", "dm")
    adapter.run_turn("telegram", session="s2", turn="t2", sender="4242")   # a real sender is still provisioned
    assert ("telegram", "4242", True) in adapter.client.resolved
    assert turns(adapter.client)[1]["sender"] == {"platform": "telegram", "user_id": "4242"}


# -- both take the same decision ----------------------------------------------------------------------------------

LABELS = ["cli", "tui", "desktop", "a2a", "bot_room", "acp", "cron", "kanban"]
IDENTITIES = [LOCAL, NO_GATEWAY, ("local", SUBJECT, "local-session", "group"), ("telegram", SUBJECT, "c", "dm"),
              ("local", "uid:1001", "local-session", "dm")]
AUTHORS = [None, OWNER_AUTHOR, BOT_AUTHOR, PEER_AUTHOR, WORKER_AUTHOR, OTHER_AUTHOR]


@pytest.mark.parametrize("task", [False, True])
def test_provider_and_capture_agree_on_every_shape(hermes, adapter, provider_mod, monkeypatch, task):
    if task:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    owners = 0
    for number, (label, bound, author) in enumerate(itertools.product(LABELS, IDENTITIES, AUTHORS)):
        session = f"s{number}"
        hermes(label=label, session=session, bound=bound, author=author)
        provider = _make_provider(provider_mod, _FakeHttpx(), monkeypatch)
        provider.initialize(session, platform=label, user_id=STARTED[0], chat_id=STARTED[1], chat_type=STARTED[2])
        provider.turn_author(author)
        provider.resolve_contact(label, SUBJECT, session, warm=False)
        info = adapter.sessions.observe(session_id=session, platform=label, sender_id=SUBJECT, turn_id="t")
        decided = provider._host_owner_turn(session)
        assert (info.host is True) is decided, (label, bound, author)
        if not task:   # a task run keeps its lane's own rules, as on Hermes 0.21; it is never captured
            assert (adapter.sessions.is_owner(session) is True) is decided, (label, bound, author)
            assert (adapter.sessions.contact_id(session) == OWNER) is decided, (label, bound, author)
        assert (provider._prefetch_contact(session) == OWNER) is decided, (label, bound, author)
        owners += decided
    # Per label, the owner's turns are: the local direct chat with no author or the owner as author (cli, tui,
    # desktop), and the terminal without a gateway (cli only). A task run has none.
    assert owners == (0 if task else 3 * 2 + 2)
    assert not [call for call in adapter.client.resolved if call[2]]


def test_provider_hook_reads_the_author_from_the_running_agent(hermes, provider_mod, monkeypatch):
    colonymind_memory = pytest.importorskip("colonymind_memory")
    fake = _FakeHttpx()
    provider = _make_provider(provider_mod, fake, monkeypatch)
    provider.initialize("s1", platform="cli", user_id=STARTED[0], chat_id=STARTED[1], chat_type=STARTED[2])
    monkeypatch.setattr(colonymind_memory, "ColonyMindMemoryProvider", type(provider))
    hooks = []
    colonymind_memory.register(types.SimpleNamespace(register_memory_provider=lambda provider: None,
                                                     register_hook=lambda name, callback: hooks.append(callback)))
    agent = sys.modules["agent.subagent_lifecycle"].get_active_subagent_parent
    for author, owner in ((BOT_AUTHOR, False), (OWNER_AUTHOR, True), (PEER_AUTHOR, False), (None, True)):
        hermes(author=author)
        agent()._memory_manager = types.SimpleNamespace(providers=[provider])
        hooks[0](user_message="Where is the spare key?", session_id="s1", platform="cli", sender_id=SUBJECT,
                 turn_id="t1")
        assert provider._host_owner_turn("s1") is owner, author
    assert not fake.requests
