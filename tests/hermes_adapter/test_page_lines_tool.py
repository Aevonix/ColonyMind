"""The agent's own page lines through its tools: ``colonymind_people boundary`` and ``colonymind_self
like|dislike``.

A line the agent writes names the turn it is written in. A session that is not the owner's own names its
person, and the sidecar decides which pages that person may open.
"""

import json
from types import SimpleNamespace

from colonymind_hermes.tools import Tools

PAGE = "Your boundaries with them:\n- #3 Keep replies short. [turn-7]"


class Sidecar:
    """Records each request and answers as the pages routes do."""

    def __init__(self):
        self.requests = []

    def has_mind_routes(self):
        return True

    def request(self, method, path, timeout=5, **sent):
        self.requests.append((method, path, sent.get("json") or sent.get("params")))
        body = {"subject": "p-02", "lines": [], "text": PAGE}
        if method == "POST":
            body = {"ok": True, "added": 3, "dropped": False, **body}
        return SimpleNamespace(is_success=True, status_code=200, json=lambda: body)


class Sessions:
    def __init__(self, owner):
        self.owner = owner

    def get(self, session_id):
        return SimpleNamespace(turn_id="turn-7", platform="sms")

    def is_owner(self, session_id):
        return self.owner

    def contact_id(self, session_id):
        return "p-01" if self.owner else "p-02"


def tools(*, owner):
    sidecar = Sidecar()
    settings = SimpleNamespace(mind=lambda: {}, owner_contact_id=lambda: "p-01", home=None)
    return Tools(sidecar, Sessions(owner), settings), sidecar


def test_a_boundary_names_the_turn_and_the_sessions_person():
    guest, sidecar = tools(owner=False)
    answer = json.loads(guest.people_tool({"operation": "boundary", "contact_id": "Casey Park",
                                           "text": "Keep replies short."}, session_id="s-1"))
    assert answer == {"ok": True, "text": PAGE, "dropped": False}
    assert json.loads(guest.people_tool({"operation": "boundary", "contact_id": "p-02"}, session_id="s-1")) \
        == {"text": PAGE}
    assert sidecar.requests == [
        ("POST", "/v1/mind/pages/Casey%20Park/lines", {"kind": "boundary", "text": "Keep replies short.", "drop": "",
                                                       "source": "turn-7", "contact_id": "p-02"}),
        ("GET", "/v1/mind/pages/p-02", {"contact_id": "p-02"})]
    # The owner's own session names no viewer: it opens every page.
    owner, sidecar = tools(owner=True)
    owner.people_tool({"operation": "boundary", "contact_id": "p-02", "drop": "3", "text": "Be brief."},
                      session_id="s-2")
    assert sidecar.requests == [("POST", "/v1/mind/pages/p-02/lines", {
        "kind": "boundary", "text": "Be brief.", "drop": "3", "source": "turn-7"})]
    assert "contact_id" in json.loads(owner.people_tool({"operation": "boundary"}, session_id="s-2"))["error"]


def test_a_taste_is_written_to_the_agents_own_page_and_dropped_by_its_id():
    guest, sidecar = tools(owner=False)
    assert json.loads(guest.self_tool({"operation": "like", "taste": "Long walks."}, session_id="s-1"))["ok"]
    guest.self_tool({"operation": "dislike", "id": "3"}, session_id="s-1")
    assert json.loads(guest.self_tool({"operation": "like"}, session_id="s-1")) == {"text": PAGE}
    assert sidecar.requests == [
        ("POST", "/v1/mind/pages/self/lines", {"kind": "like", "text": "Long walks.", "drop": "", "source": "turn-7",
                                               "contact_id": "p-02"}),
        ("POST", "/v1/mind/pages/self/lines", {"kind": "dislike", "text": "", "drop": "3", "source": "turn-7",
                                               "contact_id": "p-02"}),
        ("GET", "/v1/mind/pages/self", {"contact_id": "p-02"})]
