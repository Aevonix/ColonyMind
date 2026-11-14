"""Every plugin tool call that cannot succeed this turn gets the one final answer (``client.final_answer``).

The memory provider's system note tells the model that a ``retry: false`` answer is final for the turn. A
refusal, an id or code nobody listed, an empty search, a guest's guard block or an unreachable sidecar
answered any other way sent the model to other spellings until Hermes' iteration cap ended the turn. This
walks every failure branch of every plugin tool and of the guard in a contact's session: a branch the model
cannot fix by correcting an argument answers finally; one it can fix names the valid form and stays an
ordinary error. The source check keeps a new failure branch from arriving as an ad-hoc error.
"""

import json
import re

from conftest import ROOT, probe

CODE = '''
from tools.registry import registry
from hermes_cli.plugins import get_plugin_manager as _manager
def session(name, sender, message="hi", platform="telegram"):
    invoke_hook("pre_llm_call", session_id=name, task_id="t", turn_id="turn", user_message=message,
                conversation_history=[], is_first_turn=True, model="m", platform=platform,
                parent_session_id="", sender_id=sender)
    return name
def call(tool, args, name):
    return json.loads(registry.dispatch(tool, args, task_id="t", session_id=name))
def check(tool, args, name):
    from hermes_cli.plugins import get_pre_tool_call_directive
    action, message = get_pre_tool_call_directive(tool, args, session_id=name, task_id="t", tool_call_id="c",
                                                  turn_id="turn", api_request_id="r")
    return {"action": action, "message": message}
'''

# (label, tool, args, session) for the branches only a later turn (or the owner) can change.
FINAL = [
    ("guest log", "colonymind_self", {"operation": "log"}, "g"),
    ("guest why", "colonymind_self", {"operation": "why", "id": "i-01"}, "g"),
    ("log given content", "colonymind_self", {"operation": "log", "content": "note this"}, "o"),
    ("state given text", "colonymind_self", {"operation": "state", "text": "record: fine with it"}, "o"),
    ("opinions given a stance", "colonymind_self", {"operation": "opinions", "stance": "go with Ash"}, "o"),
    ("no operation", "colonymind_self", {"opinion": "Ash"}, "o"),
    ("why of an unknown id", "colonymind_self", {"operation": "why", "id": "p-96-venue"}, "o"),
    ("why of an unknown opinion", "colonymind_self", {"operation": "why", "id": "7"}, "o"),
    ("guest rate", "colonymind_self", {"operation": "rate", "id": "i-01", "verdict": "useful"}, "g"),
    ("guest yes", "colonymind_self", {"operation": "yes", "code": "K7F"}, "g"),
    ("guest withdraw", "colonymind_self", {"operation": "withdraw", "id": "1", "reason": "x"}, "g"),
    ("yes without a typed code", "colonymind_self", {"operation": "yes", "code": "CHECKIN98"}, "o-plain"),
    ("yes with an invented code", "colonymind_self", {"operation": "yes", "code": "P08CHK"}, "o-plain"),
    ("no without a code", "colonymind_self", {"operation": "no"}, "o-plain"),
    ("yes with the typed code", "colonymind_self", {"operation": "yes", "code": "K7F"}, "o-code"),
    ("show in a lane with no chat", "colonymind_self", {"operation": "show", "id": "1"}, "o"),
    ("guest permission", "colonymind_people", {"operation": "set_permission", "contact_id": "p-03",
                                              "permission": "auto"}, "g"),
    ("unknown person", "colonymind_people", {"operation": "inspect", "contact_id": "Nobody Here"}, "o"),
    ("search without a participant", "colonymind_memory_search", {"query": "x"}, "never-seen"),
    ("search with no hits", "colonymind_memory_search", {"query": "who is p-71"}, "g"),
    ("guest forget", "colonymind_memory_forget", {"source_ids": ["src-1"]}, "g"),
    ("guest reminder", "colonymind_reminder", {"operation": "schedule", "source_id": "s", "source_version": "v",
                                              "claim_id": "c"}, "g"),
    ("unknown reminder job", "colonymind_reminder", {"operation": "inspect", "job_id": "nope"}, "o"),
]
# (label, tool, args, session, words the error names) for the branches a corrected argument fixes.
CORRECTABLE = [
    ("device operation", "colonymind_device_invoke", {"operation": "later"}, "o",
     "use inventory, status with id, or request with invocation"),
    ("device status without id", "colonymind_device_invoke", {"operation": "status"}, "o", "status with id"),
    ("device request without invocation", "colonymind_device_invoke", {"operation": "request"}, "o",
     "request with invocation"),
    ("since_hours", "colonymind_self", {"operation": "log", "since_hours": "soon"}, "o", "number of hours"),
    ("misspelled operation", "colonymind_self", {"operation": "opinion", "query": "Ash"}, "o", "one of state, log"),
    ("why without an id", "colonymind_self", {"operation": "why"}, "o", "id is required"),
    ("show a number nobody listed", "colonymind_self", {"operation": "show", "id": "7"}, "o-chat",
     "number of an open ask"),
    ("rate without a verdict", "colonymind_self", {"operation": "rate", "id": "i-01"}, "o", "verdict"),
    ("withdraw without a number", "colonymind_self", {"operation": "withdraw", "reason": "x"}, "o", "opinion number"),
    ("withdraw without a reason", "colonymind_self", {"operation": "withdraw", "id": "1"}, "o", "reason is required"),
    ("people operation", "colonymind_people", {"operation": "delete"}, "o", "one of"),
    ("people without a name", "colonymind_people", {"operation": "inspect"}, "o", "contact_id"),
    ("guest listing", "colonymind_people", {"operation": "who"}, "g", "name who you mean"),
    ("link handle", "colonymind_people", {"operation": "propose_link", "contact_id": "p-03", "handle": "x"}, "o",
     "gateway:address"),
    ("merge without drop", "colonymind_people", {"operation": "merge", "contact_id": "p-03"}, "o", "drop"),
    ("permission value", "colonymind_people", {"operation": "set_permission", "contact_id": "p-03",
                                              "permission": "sometimes"}, "o", "never, ask or auto"),
    ("cadence minutes", "colonymind_people", {"operation": "set_cadence", "contact_id": "p-03", "minutes": "x"}, "o",
     "whole number"),
    ("search without a query", "colonymind_memory_search", {}, "o", "query is required"),
    ("forget without ids", "colonymind_memory_forget", {}, "o", "source_ids"),
    ("reminder operation", "colonymind_reminder", {"operation": "later"}, "o", "schedule, inspect or cancel"),
    ("reminder without a source", "colonymind_reminder", {"operation": "schedule"}, "o", "required"),
]


def _final(answer) -> bool:
    return answer == {"unavailable": True, "retry": False, "reason": answer.get("reason")} and bool(answer["reason"])


def test_every_tool_failure_that_cannot_succeed_this_turn_answers_finally(home, sidecar):
    sidecar.mind_routes = True
    sidecar.mind.intentions["i-01"] = {"id": "i-01", "kind": "task", "status": "asked", "ask_code": "K7F",
                                       "title": "Email the vendor"}
    original = sidecar.dispatch

    def dispatch(method, path, query, body):
        if path == "/v1/host/memory/search" and body.get("person_id") == "p-02":
            return 200, {"content": "", "count": 0, "source_refs": []}   # a first contact: nothing retained
        return original(method, path, query, body)
    sidecar.dispatch = dispatch
    result = probe(CODE + '''
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_context import set_session_vars
# The owner's message as the gateway received it: only a genuine inbound message is read as an answer.
invoke_hook("pre_gateway_dispatch", gateway=None, session_store=None, event=MessageEvent(
    text="what is waiting for me?", message_type=MessageType.TEXT, user_id="1001", message_id="m-1",
    source=SessionSource(platform=Platform("telegram"), chat_id="1001", chat_type="dm", user_id="1001")))
set_session_vars(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm", session_id="owner-chat",
                 message_id="m-1")
chat = session("owner-chat", "1001", "what is waiting for me?")
set_session_vars(platform="", user_id="", chat_id="", chat_type="", session_id="")
names = {"g": session("guest-1", "2002"), "o": session("owner-1", "1001", "what is on your list?"), "o-chat": chat,
         "o-plain": session("owner-2", "1001", "they are fine with you messaging them directly"),
         "o-code": session("owner-3", "1001", "yes K7F"), "never-seen": "never-seen"}
FINAL, CORRECTABLE = %r, %r
final = {label: call(tool, args, names[who]) for label, tool, args, who in FINAL}
# Corrections come from the owner's own live turn (the device tool refuses a senderless one first).
set_session_vars(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm", session_id="owner-1")
correctable = {label: call(tool, args, names[who]) for label, tool, args, who, _ in CORRECTABLE}
set_session_vars(platform="", user_id="", chat_id="", chat_type="", session_id="")
emit(final=final, correctable=correctable,
     guard={"search": check("session_search", {"query": "x"}, names["g"]),
            "cron": check("cronjob_manage", {"action": "create", "schedule": "in 1h", "prompt": "hi",
                                             "deliver": "telegram:2002"}, names["g"])})
''' % (FINAL, CORRECTABLE), home)
    for label, answer in result["final"].items():
        assert _final(answer), (label, answer)
    for label, _tool, _args, _who, words in CORRECTABLE:
        answer = result["correctable"][label]
        assert "retry" not in answer and words in answer.get("error", ""), (label, answer)
    for label, verdict in result["guard"].items():
        assert verdict["action"] == "block" and verdict["message"].endswith("(retry: false)"), (label, verdict)
        assert "BLOCKED" not in verdict["message"], (label, verdict)
    # No tool call answers an ask: nothing reached the sidecar as an answer (only the owner's message is read).
    assert sidecar.calls("/v1/mind/decide", "POST") == []
    assert [call["json"]["turn_id"] for call in sidecar.calls("/v1/mind/answers", "POST")] == ["turn"]
    assert "no ask is open" not in json.dumps(result["final"])


def test_no_ask_open_and_an_unreachable_sidecar_answer_finally(home, sidecar):
    sidecar.mind_routes = True
    result = probe(CODE + '''
o = session("owner-1", "1001", "yes K7F")
emit(no_ask=call("colonymind_self", {"operation": "yes", "code": "K7F"}, o))
''', home)
    assert _final(result["no_ask"]) and "owner's own message" in result["no_ask"]["reason"]
    home.write_config(**{**home.config, "plugins": {**home.config["plugins"], "colonymind": {
        "sidecar_url": "http://127.0.0.1:1", "key_file": str(home.key_file)}}})
    down = probe(CODE + '''
o = session("owner-1", "", "hi", platform="cli")
emit(log=call("colonymind_self", {"operation": "log"}, o),
     why=call("colonymind_self", {"operation": "why", "id": "i-01"}, o),
     people=call("colonymind_people", {"operation": "who"}, o),
     search=call("colonymind_memory_search", {"query": "x"}, o),
     forget=call("colonymind_memory_forget", {"source_ids": ["src-1"]}, o),
     reminder=call("colonymind_reminder", {"operation": "schedule", "source_id": "s", "source_version": "v",
                                          "claim_id": "c"}, o))
''', home)
    for label, answer in down.items():
        assert _final(answer), (label, answer)


def test_the_provider_answers_the_same_way(monkeypatch, tmp_path):
    """The memory provider loads on its own and keeps its own copy of the answer; it is the same answer."""
    from colonymind_hermes.client import final_answer
    from colonymind_memory import provider

    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "instance"))
    assert provider._terminal("no open commitment") == final_answer("no open commitment")
    unknown = json.loads(provider.ColonyMindMemoryProvider({"contact_id": "p-01"}).handle_tool_call(
        "colonymind_nope", {}))
    assert _final(unknown), unknown


def test_no_failure_branch_is_an_ad_hoc_error():
    """Each ordinary error left in the tools is an argument the model can correct, and says how; a new branch
    must be classified (and walked above) before it can ship."""
    correctable = {"since_hours is a number of hours", "id is required", "id and verdict (",
                   "id is required (an opinion number from opinions)", "reason is required: the owner's own words",
                   "id is the number of an open ask, from asks", "operation is one of ",
                   "contact_id (a name, handle or id)",
                   "only the owner can list contacts; name who you mean", "handle is gateway:address",
                   "drop (folded into contact_id) is required", "permission is never, ask or auto",
                   "minutes is a whole number", "query is required", "source_ids is required",
                   "use schedule, inspect or cancel", "is required to schedule", "lead_seconds must be",
                   "the reminder time has already passed", "the deadline has no timezone",
                   "grant is required: {id, class, scope} and any limits", "grant is malformed: ",
                   "an action is called with its declared arguments (",
                   "handles must be a non-empty list of branch handles from Branches",
                   "question must be non-empty text",
                   "use inventory, status with id, or request with invocation"}
    for module in ("tools", "reminders"):
        source = (ROOT / f"plugins/hermes-plugin/{module}.py").read_text()
        for message in re.findall(r'(?:_error|ValueError)\(f?"([^"]*)"', source):
            assert any(known in message or message in known for known in correctable), (module, message)


def test_the_record_is_read_and_recording_is_automatic():
    """What the model reads about the record says it only reads: the pilots' model called ``log`` (and ``why``)
    with the owner's statement or its own stance to "record" it, read an empty log as a failed write and tried
    again. "Not in the log means it did not happen" is about the agent's own actions, never what a turn said,
    and the log's notes (the nightly consolidation) are not actions to cite. The record holds what the mind did or
    decided on its own: work done in a conversation is not in it (its tool results and session search hold it), so
    the description never calls the record the only source or says that work did not happen."""
    from colonymind_hermes.tools import SELF_SCHEMA

    text = SELF_SCHEMA["description"]
    assert text.startswith("What your mind did or decided on its own")
    assert "work done in a conversation is not in it" in text and "session search hold that" in text
    assert "log (read-only" in text and "cite ids of actions, not notes" in text
    assert "an action of your mind that is not there did not happen" in text
    assert "one not there did not happen" not in text and "the only source" not in text
    assert "recorded after it, with no tool call" in text
    assert "not in the log means it did not happen" not in text
    assert "grants" in text and "revoke <id>" in text


def test_no_hits_is_final_only_when_no_part_of_the_search_failed(home, sidecar):
    """Zero hits is the answer when the search ran whole: semantic recall ready, or not part of this install
    (lexical is then the whole search). When semantic recall failed (an embedding service down) only exact
    words were matched, so the answer says so and leaves the model free to search in other words: for the
    owner, a final "nothing retained" would deny a topic the memory holds."""
    statuses = {"lease agreement": "failed", "lease terms": "ready", "lease dates": "unavailable"}
    original = sidecar.dispatch

    def dispatch(method, path, query, body):
        if path == "/v1/host/memory/search":
            return 200, {"content": "", "count": 0, "source_refs": [], "watermark": 0, "annotation_checks": [],
                         "retrieval": {"semantic": statuses[body["query"]], "contact_facts": "ready"}}
        return original(method, path, query, body)
    sidecar.dispatch = dispatch
    result = probe(CODE + '''
o = session("owner-1", "1001", "what did I say about the lease last month?")
emit(**{query: call("colonymind_memory_search", {"query": query}, o) for query in %r})
''' % list(statuses), home)
    degraded = result["lease agreement"]
    assert "retry" not in degraded and degraded["count"] == 0 and degraded["content"] == ""
    assert "only exact words" in degraded["note"] and "other words" in degraded["note"]
    assert _final(result["lease terms"]) and _final(result["lease dates"])


def test_a_read_given_a_stray_argument_is_served_with_the_note(home, sidecar):
    """An argument the read does not take but the schema offers (a ``reason`` on why, a ``verdict`` on
    opinions) is ignored: the read is served, with the note that recording is automatic. Only content the
    schema has no place for (``content``, ``text``, a ``stance``) is an attempt to write and gets the final
    answer, so the model never loses a read it can use."""
    sidecar.mind_routes = True
    sidecar.mind.intentions["i-01"] = {"id": "i-01", "kind": "task", "status": "done", "title": "Email the vendor"}
    result = probe(CODE + '''
o = session("owner-1", "1001", "why did you email the vendor?")
emit(why=call("colonymind_self", {"operation": "why", "id": "i-01", "reason": "the owner asked why"}, o),
     plain=call("colonymind_self", {"operation": "why", "id": "i-01"}, o),
     log=call("colonymind_self", {"operation": "log", "verdict": "useful"}, o),
     written=call("colonymind_self", {"operation": "why", "id": "i-01", "content": "I emailed them"}, o))
''', home)
    note = result["why"].pop("note")
    assert "recorded after" in note and result["why"] == result["plain"] and "retry" not in result["why"]
    assert "recorded after" in result["log"]["note"] and "retry" not in result["log"]
    assert _final(result["written"]) and "only reads" in result["written"]["reason"]
