"""What a tool reads in a turn in a group chat.

The reply of such a turn goes to everyone in the chat, so for every sender, the owner included, a tool reads that
chat's own history only. A memory search and a page name the chat, and the sidecar narrows them; ``who`` and
``inspect`` say who one named person is; a read with no history of its own, and ``session_search``, are refused.
Each such result carries the one line that says so. A direct chat is unchanged.
"""

import json
import sys
import types
from types import SimpleNamespace

from colonymind_hermes import capture, guard
from colonymind_hermes.tools import Tools
from test_page_lines_tool import PAGE, Sessions
from test_page_lines_tool import Sidecar as Pages
from test_unknown_standing import lane  # noqa: F401  (a guard over a session map that knows one owner handle)

GROUP = {"chat_type": "group", "chat": "signal:crew"}
WITHHELD = ("this is a group chat: what does not come from this chat's own history was withheld; the owner can ask "
            "you in a direct chat")
REFUSED = {"unavailable": True, "retry": False, "reason": WITHHELD}
PUBLIC = {"contact_id": "p-02", "display_name": "Casey Park", "trust_tier": "regular"}
RECORD = {**PUBLIC, "digest": "Sails on Sundays.", "handles": [{"gateway": "signal", "address": "casey"}]}
SEVERAL = "no single contact matches 'Casey': Casey Park (regular)"


class Sidecar(Pages):
    """Answers the people routes and the memory search too."""

    def __init__(self, found=1):
        super().__init__()
        self.found = found

    def request(self, method, path, timeout=5, **sent):
        if not path.startswith("/v1/mind/people"):
            return super().request(method, path, timeout, **sent)
        self.requests.append((method, path, sent.get("json") or sent.get("params")))
        if method == "POST":     # a change whose name is nobody's exactly: the owner is told who it could mean
            body = {"detail": {"code": "unknown_contact", "candidates": [PUBLIC], "message": SEVERAL}}
            return SimpleNamespace(is_success=False, status_code=404, json=lambda: body)
        body = {"contacts": [RECORD]} if path == "/v1/mind/people" else {"contact": RECORD}
        return SimpleNamespace(is_success=True, status_code=200, json=lambda: body)

    def post(self, path, timeout=5, json=None):
        self.requests.append(("POST", path, json))
        body = {"content": "The ferry leaves at seven.", "count": self.found, "source_refs": [], "retrieval": {}}
        return SimpleNamespace(is_success=True, status_code=200, json=lambda: body)


def tools(*, owner=True, found=1):
    sidecar = Sidecar(found)
    settings = SimpleNamespace(mind=lambda: {}, owner_contact_id=lambda: "p-01", home=None)
    return Tools(sidecar, Sessions(owner), settings), sidecar


def in_group(monkeypatch):
    monkeypatch.setattr("colonymind_hermes.tools.group_chat", lambda: dict(GROUP))


def test_a_memory_search_names_the_group_chat_and_says_what_was_withheld(monkeypatch):
    owner, sidecar = tools()
    request = {"identity": {"host_id": "hermes"}, "person_id": "p-01", "session_id": "s-2", "query": "ferry",
               "limit": 5}
    found = {"content": "The ferry leaves at seven.", "count": 1, "source_refs": []}
    # In a direct chat the search is the one it was.
    assert json.loads(owner.memory_search({"query": "ferry"}, session_id="s-2")) == found
    in_group(monkeypatch)
    assert json.loads(owner.memory_search({"query": "ferry"}, session_id="s-2")) == {**found, "narrowed": WITHHELD}
    assert sidecar.requests == [("POST", "/v1/host/memory/search", request),
                                ("POST", "/v1/host/memory/search", {**request, **GROUP})]
    nothing, _ = tools(found=0)
    assert json.loads(nothing.memory_search({"query": "picnic"}, session_id="s-2")) == {
        **REFUSED, "reason": "nothing retained matches; " + WITHHELD}


def test_a_page_read_and_a_page_write_name_the_group_chat_and_the_session(monkeypatch):
    in_group(monkeypatch)
    owner, sidecar = tools()
    assert json.loads(owner.self_tool({"operation": "like"}, session_id="s-2")) == {
        "text": PAGE, "narrowed": WITHHELD}
    guest, visited = tools(owner=False)
    written = json.loads(guest.people_tool({"operation": "boundary", "contact_id": "p-02",
                                            "text": "Keep replies short."}, session_id="s-1"))
    assert written == {"ok": True, "text": PAGE, "dropped": False, "narrowed": WITHHELD}
    assert sidecar.requests == [("GET", "/v1/mind/pages/self", {"session_id": "s-2", **GROUP})]
    assert visited.requests == [("POST", "/v1/mind/pages/p-02/lines", {
        "kind": "boundary", "text": "Keep replies short.", "drop": "", "source": "turn-7", "contact_id": "p-02",
        "session_id": "s-1", **GROUP})]


def test_the_owner_in_a_group_chat_reads_who_one_named_person_is_and_no_record(monkeypatch):
    owner, sidecar = tools()
    inspect, cadence = {"operation": "inspect", "contact_id": "p-02"}, {
        "operation": "set_cadence", "contact_id": "Casey", "minutes": 60}
    # In the owner's direct chat the record is shown, and a name that is nobody's exactly says who it could mean.
    assert json.loads(owner.people_tool(inspect, session_id="s-2")) == {
        **PUBLIC, "digest": "Sails on Sundays.", "handles": ["signal:casey"]}
    assert json.loads(owner.people_tool(cadence, session_id="s-2"))["reason"] == SEVERAL
    in_group(monkeypatch)
    assert json.loads(owner.people_tool(inspect, session_id="s-2")) == {**PUBLIC, "narrowed": WITHHELD}
    assert json.loads(owner.people_tool({"operation": "who", "contact_id": "Casey"}, session_id="s-2")) == {
        "result": [PUBLIC], "narrowed": WITHHELD}
    assert json.loads(owner.people_tool(cadence, session_id="s-2")) == {
        **REFUSED, "reason": "no single contact matches; " + WITHHELD}
    # The two reads ask the sidecar as a participant who is not the owner does: it lists nobody and shows no record.
    assert [sent for method, _, sent in sidecar.requests if method == "GET"] == [
        {}, {"contact_id": ""}, {"q": "Casey", "limit": 10, "contact_id": ""}]


def test_a_read_with_no_history_of_its_own_is_refused_in_a_group_chat(monkeypatch):
    in_group(monkeypatch)
    owner, sidecar = tools()
    for call in ({"operation": "opinions"}, {"operation": "why", "id": "4"},
                 {"operation": "rate", "id": "i-1", "verdict": "useful"}):
        assert json.loads(owner.self_tool(call, session_id="s-2")) == REFUSED, call
    assert sidecar.requests == []


def test_session_search_is_refused_in_a_group_chat_to_the_owner_and_a_delegate(lane, monkeypatch):  # noqa: F811
    gate, sessions = lane
    sessions.observe(session_id="chat-1", platform="telegram", sender_id="owner-1")
    gate.subagent_start(child_session_id="child-1", parent_session_id="chat-1")
    sessions.observe(session_id="child-1", platform="subagent", parent_session_id="chat-1")
    # In a direct chat the owner's search of other sessions is not the guard's to stop (``decide`` is the guard's
    # own verdict; the hook around it also registers an allowed call with the host's tool registry).
    assert gate.decide("session_search", {"query": "x"}, "chat-1") is None
    monkeypatch.setattr(guard, "group_chat", lambda: {"chat_type": "group", "chat": "telegram:-1007"})
    for session in ("chat-1", "child-1"):
        assert gate.pre_tool_call("session_search", {"query": "x"}, session) == {
            "action": "block", "message": WITHHELD + " (retry: false)"}, session
    # The owner's other tools are as they were.
    assert gate.decide("terminal", {}, "chat-1") is None and gate.decide("terminal", {}, "child-1") is None


def test_the_turns_group_chat_is_the_one_the_gateway_bound(monkeypatch):
    bound = {}
    context = types.ModuleType("gateway.session_context")
    context.get_session_env = lambda name, default="": bound.get(name, default)
    package = types.ModuleType("gateway")
    package.session_context = context
    monkeypatch.setitem(sys.modules, "gateway", package)
    monkeypatch.setitem(sys.modules, "gateway.session_context", context)
    # A chat kind that is not a direct one names the chat as the memory provider names it for the turn's recall.
    for kind, chat, named in (("Group", "-1007", {"chat_type": "group", "chat": "telegram:-1007"}),
                              ("channel", "", {"chat_type": "channel"}), ("dm", "casey", {}),
                              ("direct", "casey", {}), ("private", "casey", {}), ("", "", {})):
        bound.update(HERMES_SESSION_PLATFORM="Telegram", HERMES_SESSION_CHAT_ID=chat, HERMES_SESSION_CHAT_TYPE=kind)
        assert capture.group_chat() == named, kind
    monkeypatch.setitem(sys.modules, "gateway.session_context", None)     # no gateway behind the turn
    assert capture.group_chat() == {}
    assert capture.GROUP_WITHHELD == WITHHELD
