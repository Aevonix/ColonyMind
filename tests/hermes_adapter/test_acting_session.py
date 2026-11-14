"""The adapter names the session it acts for in ``X-ColonyMind-Session``, so the sidecar can refuse a hand session on
routes whose request body names none (the answer route, people changes, memory writes)."""

from types import SimpleNamespace

from colonymind_hermes.answers import Answers
from colonymind_hermes.client import ACTING_SESSION, ColonyMindClient, acting
from colonymind_hermes.tools import Tools, in_session


def test_no_session_no_header():
    assert "X-ColonyMind-Session" not in ColonyMindClient(url="http://127.0.0.1:9", api_key="k").headers()


def test_acting_names_the_session_only_inside_the_call():
    c = ColonyMindClient(url="http://127.0.0.1:9", api_key="k")
    inside = acting("child-session", c.headers)
    assert inside["X-ColonyMind-Session"] == "child-session" and inside["Authorization"] == "Bearer k"
    assert ACTING_SESSION.get() == "" and "X-ColonyMind-Session" not in c.headers()


def test_every_tool_runs_in_its_session(monkeypatch):
    names = ("self_tool", "people_tool", "memory_search", "memory_forget", "action_tool", "consult", "device_tool")
    assert all(getattr(Tools, name).__wrapped__ for name in names)
    probe = in_session(lambda self, args=None, *, session_id="", **_: ACTING_SESSION.get())
    assert probe(None, {}, session_id="hand-child-1", task_id="t") == "hand-child-1"
    assert ACTING_SESSION.get() == ""
    settings = SimpleNamespace(mind=lambda: {}, owner_contact_id=lambda: "owner", home=None)
    monkeypatch.setattr(Tools, "_actions", lambda self: [{"name": "fixture"}])
    monkeypatch.setattr("colonymind_hermes.tools.action_schema", lambda actions: {"name": "fixture_action"})
    monkeypatch.setattr("colonymind_hermes.tools.branches_enabled", lambda *a, **kw: True)
    handlers = Tools(ColonyMindClient(url="http://127.0.0.1:9", api_key="k"), None, settings).handlers()
    assert sorted(handler.__func__.__name__ for _, handler in handlers) == sorted(names)
    assert all(handler.__func__.__wrapped__ for _, handler in handlers)


def test_the_answer_read_names_the_owner_session(monkeypatch):
    seen = []
    answers = object.__new__(Answers)

    def read(session_id, turn_id):
        seen.append(ColonyMindClient(url="http://127.0.0.1:9", api_key="k").headers().get("X-ColonyMind-Session"))
        return None

    answers._pre_llm_call = read
    assert answers.pre_llm_call(session_id="owner-session", turn_id="turn-1") is None
    assert seen == ["owner-session"] and ACTING_SESSION.get() == ""
