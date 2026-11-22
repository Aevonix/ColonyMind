"""Model tools: ``colonymind_self``, ``colonymind_people`` and the memory tools.

Tool handlers receive ``session_id``; the session map turns it into a sender.
Reads of the mind and the contact list are the owner's and the owner's own
senderless lanes', never a guest's. Mutations are accepted only from the
owner's own interactive session: never
from a kanban worker, and never from a cron run, whose prompt is a stored job
anyone with the tool could have scheduled. No tool answers an ask: the owner's
own message is read before the model runs (``answers.py``, docs/MIND.md,
Asks), so neither a guest, a worker, a cron job, an injected page nor the model
can approve anything; ``asks`` lists the open asks in words and ``show`` puts
one in front of the owner again as its own message (architecture 7.7, 7.10).
The mind's record (what it works on, its log, why it acted, the
self-narrative) is the owner's too: shown only in the owner's own session,
never to a guest, a group the owner shares or a worker. A call that cannot succeed this turn (a refusal,
an id or code nobody listed, nothing retained, an unreachable sidecar) gets ``final_answer``, never an
error the model would try again with other arguments; an argument it can correct names the valid form.

``colonymind_action`` runs one of the owner's declared actions (``mind.actions``: a fixed command, run without a
shell, whose only free words are checked arguments); it is offered only when the owner declared one, runs for the
owner's lanes and a mind run the guard let through, and never for a guest.
"""

from __future__ import annotations

from datetime import datetime, timezone
import functools
import json
import os
import subprocess
from typing import Any, Callable
from urllib.parse import quote

from .answers import Answers
from .body import mind_state
from .capture import SessionMap, branch_owner, branch_sender, branches_enabled, session_env
from .client import ColonyMindClient, Settings, SidecarUnavailable, acting, final_answer
from .commands import ROUTES_MISSING
from .guard import ACTION_TIMEOUT, action_argv, action_problem, action_values

RECORD_IS_OWNERS = "the mind's record is the owner's and is shown only in the owner's own chat"
# What a read of the record takes. Another argument the schema offers (a reason, a verdict) is ignored and the read
# served with the note that recording is automatic; content the schema has no place for (content, text, a stance)
# is an attempt to write through it. The record is written after the turn, from what it said, never by a call.
READ_KEYS = frozenset({"operation", "id", "limit", "since_hours", "kind", "recipient", "query"})
RECORDED = "what this turn says is recorded after it, with no tool call"
READS_ONLY = f"colonymind_self only reads the record; {RECORDED}"

DEVICE_SCHEMA = {
    "name": "colonymind_device_invoke",
    "description": "List registered iPhone Health, alarms and Home capabilities, request one invocation, or read its result. "
                   "An OS grant and agent authorization are separate. Ask mode waits for the owner's separate approval. "
                   "Use the same invocation id after an uncertain request; a timeout never warrants a new automatic attempt.",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": ["inventory", "request", "status"]},
        "id": {"type": "string", "description": "Invocation id for status."},
        "invocation": {"type": "object", "properties": {
            "id": {"type": "string", "description": "Unique stable request id; retain it when checking an uncertain request."},
            "device_id": {"type": "string"},
            "capability": {"type": "string", "enum": ["health", "alarms", "home"]},
            "command": {"type": "string", "description": "Exact command advertised by inventory."},
            "parameters": {"type": "object"},
            "deadline": {"type": "string", "description": "ISO timestamp with timezone, within the next hour."},
        }, "required": ["id", "device_id", "capability", "command", "deadline"], "additionalProperties": False},
    }, "required": ["operation"], "additionalProperties": False},
}

VERDICTS = ("actioned", "dismissed", "ignored", "useful", "not_useful", "wrong")
# Turns nobody typed: stock cron runs its agents with ``platform="cron"`` and no sender.
AUTOMATED_PLATFORMS = frozenset({"cron"})
PEOPLE_OPERATIONS = ("who", "inspect", "propose_link", "merge", "set_permission", "set_cadence", "boundary")
OWNER_PEOPLE_OPERATIONS = frozenset({"merge", "set_permission", "set_cadence"})
# A person as the model sees it: who they are to everyone; permission, cadence and handles to the owner.
PUBLIC_PERSON = ("contact_id", "display_name", "trust_tier")
OWNER_PERSON = PUBLIC_PERSON + ("may_contact", "cadence_minutes", "last_interaction_at", "digest", "handles")

# Every schema is sent with every model request, so each says only what the model needs to pick
# the tool and fill its arguments; the handlers validate and explain the rest.
# M7's opinions ride this tool (integration map X8): opinions [query], why <opinion number>, and the owner's
# withdraw|reconsider <number> with a reason; the sidecar filters every read by the session's participant.
# The owner's standing grants ride it too: ``grant`` files one the owner asked for in their own words as an ask
# they confirm with their own yes, and ``revoke`` narrows at once (docs/MIND.md, "Standing grants"). No operation
# answers an ask: ``asks`` lists them in words and ``show`` puts one in front of the owner as its own message.
SELF_OPERATIONS = ("state", "log", "why", "rate", "asks", "show", "opinions", "withdraw", "reconsider", "grant",
                   "revoke", "like", "dislike")
# The operations a model may remember from before: an ask is answered in the owner's own message, never a tool call.
ANSWER_OPERATIONS = frozenset({"yes", "no"})
SELF_SCHEMA = {
    "name": "colonymind_self",
    "description": "What your mind did or decided on its own (work done in a conversation is not in it: its tool "
                   "results and session search hold that): state (level, working_on, narrative, grants), log "
                   "(read-only; cite ids of actions, not notes; an action of your mind that is not there did not "
                   "happen), why <id or opinion number>, rate <id> <verdict>, "
                   "opinions [query], like|dislike [taste] (your own tastes; id drops one); "
                   "owner only: asks, show <n> (sends ask n to the owner after your reply; they "
                   "answer in their own words), withdraw|reconsider <opinion> <reason>, grant <grant> <reason: the "
                   "owner's words> (active on the owner's yes), revoke <id>. What a turn says is recorded after it, "
                   "with no tool call.",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": list(SELF_OPERATIONS)},
        "id": {"type": "string"}, "verdict": {"type": "string", "enum": list(VERDICTS)},
        "limit": {"type": "integer"}, "since_hours": {"type": "number"},
        "kind": {"type": "string"}, "recipient": {"type": "string"}, "query": {"type": "string"},
        "reason": {"type": "string"}, "taste": {"type": "string"},
        "grant": {"type": "object", "description": "{id, class, scope, tools: actions or structured tools} and "
                                                   "limits: match, max_amount, currency, per_day, per_month_amount, "
                                                   "expires"}},
        "required": ["operation"]}}
PEOPLE_SCHEMA = {
    "name": "colonymind_people",
    "description": "contact_id: a name, handle or id. propose_link: handle (gateway:address) is contact_id; owner "
                   "confirms. boundary [text]: how you treat contact_id, kept on your page about them (drop: line id "
                   "to remove). Owner only: set_permission, set_cadence (minutes, 0 clears), merge (drop into "
                   "contact_id).",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": list(PEOPLE_OPERATIONS)}, "contact_id": {"type": "string"},
        "drop": {"type": "string"}, "handle": {"type": "string"}, "text": {"type": "string"},
        "permission": {"type": "string", "enum": ["never", "ask", "auto"]}, "minutes": {"type": "integer"}},
        "required": ["operation"]}}
# With mind.faculties.people false (the full-people ablation) the tool keeps its pre-M5 surface.
M4_PEOPLE_OPERATIONS = ("who", "inspect", "set_permission")
M4_PEOPLE_SCHEMA = {
    "name": "colonymind_people", "description": "contact_id: a name, handle or id. Owner only: set_permission.",
    "parameters": {"type": "object", "required": ["operation"], "properties": {
        "operation": {"type": "string", "enum": list(M4_PEOPLE_OPERATIONS)}, "contact_id": {"type": "string"},
        "permission": {"type": "string", "enum": ["never", "ask", "auto"]}}}}
# The search's own time (docs/RECALL-HYBRID.md, "Recall and time"): ``window`` is the span the question asks about,
# whatever its words say, so a question in a language the sidecar's date parser does not read is not held to English
# date words; ``as_of`` is the time its relative dates count from, not a view of what was believed then.
SEARCH_SCHEMA = {
    "name": "colonymind_memory_search",
    "description": "Search this participant's memory beyond this turn's recall. window {start, end}: the span asked "
                   "about; as_of: the time relative dates count from; ISO with offset.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                                                    "window": {"type": "object"}, "as_of": {"type": "string"}},
                   "required": ["query"]}}
FORGET_SCHEMA = {
    "name": "colonymind_memory_forget",
    "description": "Owner only: forget retained sources by exact source_id when the owner asks.",
    "parameters": {"type": "object", "properties": {"source_ids": {"type": "array", "items": {"type": "string"}}},
                   "required": ["source_ids"]}}
CONSULT_SCHEMA = {
    "name": "colonymind_consult",
    "description": "Ask a question of retained branch material. Use handles from Branches; multiple handles are "
                   "read together. The answer cites span handles. Owner's private session only.",
    "parameters": {"type": "object", "properties": {
        "handles": {"type": "array", "items": {"type": "string"}}, "question": {"type": "string"}},
        "required": ["handles", "question"]}}


def action_schema(actions: dict[str, Any]) -> dict[str, Any]:
    """``colonymind_action`` for the owner's declared actions: each one's name, what it does and its arguments."""
    listed = "; ".join(
        f"{name}: {spec['description'].strip()} ("
        + (", ".join(f"{arg} {check.get('enum') or check.get('pattern') or check['type']}"
                     for arg, check in (spec.get("args") or {}).items()) or "no args") + ")"
        for name, spec in actions.items())
    return {"name": "colonymind_action",
            "description": f"Run one of the owner's declared actions (a fixed command; args are checked). {listed}",
            "parameters": {"type": "object", "required": ["action"], "properties": {
                "action": {"type": "string", "enum": list(actions)}, "args": {"type": "object"}}}}


# A forget answers in about a second on a long history (the closure is indexed; the
# vector tables are compacted routinely). Past this bound the tool says the removal
# is unconfirmed, never that it failed: the sidecar finishes a forget it received.
FORGET_TIMEOUT_SECONDS = 5.0


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _error(message: str) -> str:
    return _json({"error": message})


TIME_FORM = "an ISO 8601 time with a UTC offset, such as 2026-03-02T00:00:00+01:00"


def _instant(value: Any) -> str | None:
    """``value`` as the sidecar reads a time, in UTC: an ISO 8601 string with a UTC offset that names an instant;
    None for anything else (a local date or time the sidecar cannot place, words, a number)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        stamp = datetime.fromisoformat(value.strip())
        return None if stamp.utcoffset() is None else stamp.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError):
        return None


def search_time(args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The search's ``window`` and ``as_of`` as ``POST /v1/host/memory/search`` takes them, or what the model must
    correct. The sidecar refuses a naive time and a window whose end is not after its start (422); such a value is
    never sent, and a field the model left out or empty is not sent either."""
    fields: dict[str, Any] = {}
    window = args.get("window")
    if window not in (None, "", {}):
        bounds = window if isinstance(window, dict) else {}
        start, end = _instant(bounds.get("start")), _instant(bounds.get("end"))
        if start is None or end is None:
            return {}, f"window is {{start, end}}, each {TIME_FORM}; the end is exclusive"
        if datetime.fromisoformat(end) <= datetime.fromisoformat(start):
            return {}, "window ends after it starts: the end is exclusive, so a day's window ends at the next midnight"
        fields["window"] = {"start": start, "end": end}
    if args.get("as_of") not in (None, ""):
        as_of = _instant(args["as_of"])
        if as_of is None:
            return {}, f"as_of is {TIME_FORM}: the time relative dates count from"
        fields["as_of"] = as_of
    return fields, None


def _opinion(value: Any) -> int | None:
    """An opinion number (an intention id is a UUID, never all digits)."""
    text = "" if isinstance(value, bool) or not isinstance(value, (int, str)) else str(value).strip()
    return int(text) if text.isdigit() else None


def _noted(answer: str) -> str:
    """A served read, with the note that what the turn says is recorded after it; a refusal is left as it is."""
    try:
        value = json.loads(answer)
    except (ValueError, RecursionError):
        value = answer
    if isinstance(value, dict) and ("error" in value or value.get("unavailable") is True):
        return answer
    return _json({**value, "note": RECORDED} if isinstance(value, dict) else {"result": value, "note": RECORDED})


# What the model reads of a change the sidecar filed as an ask (a standing grant, a widening people change): its
# code is never the model's to show, since the owner answers the ask's own message in their own words.
ASKED = "not changed yet: the owner is asked as its own message and their own yes applies it; do not ask them for a code"


def _without_code(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("status") != "asked":
        return value
    return {**{key: item for key, item in value.items() if key not in {"code", "ask_code"}}, "text": ASKED}


def in_session(method: Callable[..., str]) -> Callable[..., str]:
    """A tool's calls to the sidecar name the session it runs in (``client.acting``)."""
    @functools.wraps(method)
    def run(self: Any, args: Any = None, *, session_id: str = "", **kwargs: Any) -> str:
        return acting(session_id, method, self, args, session_id=session_id, **kwargs)
    return run


class Tools:
    def __init__(self, client: ColonyMindClient, sessions: SessionMap, settings: Settings,
                 answers: Answers | None = None):
        self.client, self.sessions, self.settings, self.answers = client, sessions, settings, answers
        faculties = settings.mind().get("faculties")
        self.people_on = not isinstance(faculties, dict) or faculties.get("people") is not False

    def handlers(self) -> list[tuple[dict[str, Any], Callable[..., str]]]:
        actions = self._actions()
        return [(SELF_SCHEMA, self.self_tool), (PEOPLE_SCHEMA if self.people_on else M4_PEOPLE_SCHEMA, self.people_tool),
                (SEARCH_SCHEMA, self.memory_search), (FORGET_SCHEMA, self.memory_forget),
                (DEVICE_SCHEMA, self.device_tool)] + \
            ([(action_schema(actions), self.action_tool)] if actions else []) \
            + ([(CONSULT_SCHEMA, self.consult)] if branches_enabled(self.settings, consult=True) else [])

    def consult_available(self) -> bool:
        """The stock tool registry's availability check also respects automatic withdrawal."""
        if not branches_enabled(self.settings, consult=True) or os.environ.get("HERMES_KANBAN_TASK"):
            return False
        platform, sender, _chat, chat_type = session_env()
        if chat_type not in {"", "dm", "direct", "private"}:
            return False
        try:
            response = self.client.get("/v1/memory/branches", timeout=2, params={
                "session_id": "", "requester": self.settings.owner_contact_id(),
                "platform": platform or "cli", "user_id": sender})
            return response.is_success and response.json().get("tool_available") is True
        except (SidecarUnavailable, ValueError, AttributeError):
            return False

    @in_session
    def consult(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        if not branches_enabled(self.settings, consult=True):
            return final_answer("working-memory consults are disabled")
        if not self._owners_own(session_id):
            return final_answer("branches are available only in the owner's private session")
        owner = branch_owner(self.sessions, session_id)
        if owner is None:
            return final_answer("this session has no verified owner binding")
        args = args if isinstance(args, dict) else {}
        handles, question = args.get("handles"), args.get("question")
        if not isinstance(handles, list) or not handles or not all(isinstance(h, str) and h for h in handles):
            return _error("handles must be a non-empty list of branch handles from Branches")
        if not isinstance(question, str) or not question.strip():
            return _error("question must be non-empty text")
        try:
            # Assumption: a bounded two-minute consult, including a cold read of retained text.
            response = self.client.post("/v1/memory/branches/consult", timeout=120, json={
                "handles": handles, "question": question, "session_id": session_id,
                "requester": self.settings.owner_contact_id(), "sender": branch_sender(owner)})
        except SidecarUnavailable:
            return final_answer("the sidecar is unreachable")
        if not response.is_success:
            try:
                detail = response.json().get("detail")
                reason = (detail.get("message") or detail.get("code")) if isinstance(detail, dict) else detail
                if isinstance(detail, dict) and detail.get("handles"):
                    reason = f"{reason or 'consult refused'}: {', '.join(map(str, detail['handles']))}"
            except (ValueError, AttributeError):
                reason = ""
            return final_answer(str(reason or f"consult refused: HTTP {response.status_code}"))
        info = self.sessions.get(session_id)
        if info is not None:
            info.branch_consult_turn = info.turn_id
        return response.text

    # -- helpers ------------------------------------------------------------------

    def _owner(self, session_id: str) -> bool:
        """The owner's own interactive session: never a kanban worker, whose turns carry no
        sender, and never a cron run, whose prompt is a stored job rather than a typed message."""
        if os.environ.get("HERMES_KANBAN_TASK"):
            return False
        info = self.sessions.get(session_id)
        if info is not None and info.platform.strip().lower() in AUTOMATED_PLATFORMS:
            return False
        return self.sessions.is_owner(session_id) is True

    def _owners_own(self, session_id: str) -> bool:
        """The owner's session with no one else in it (a direct chat, the CLI): the mind's record (what it
        works on, whom it messaged, what it knows about itself) is shown there only, never to a guest, in a
        group the owner shares, or to a worker."""
        return self._owner(session_id) and session_env()[3].lower() in {"", "dm"}

    def _mind(self, method: str, path: str, **kwargs: Any) -> str:
        if self.client.has_mind_routes() is not True:
            return final_answer(ROUTES_MISSING)
        try:
            response = self.client.request(method, path, timeout=5, **kwargs)
        except SidecarUnavailable:
            return final_answer("the sidecar is unreachable")
        if response.status_code == 404:
            return final_answer(ROUTES_MISSING if not response.text else f"not found: {response.text[:200]}")
        if response.status_code == 403:     # a refusal says why (an answer that does not answer the ask)
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            return final_answer(str(detail.get("message") if isinstance(detail, dict) else "") or "refused")
        if not response.is_success:
            try:  # the sidecar's reason, when it gives one
                reason = response.json()["detail"]["message"]
            except Exception:
                reason = None
            return final_answer(str(reason) if reason else f"sidecar HTTP {response.status_code}")
        return response.text

    # -- colonymind_self ---------------------------------------------------------------

    @in_session
    def self_tool(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        args = args if isinstance(args, dict) else {}
        operation = str(args.get("operation") or "")
        if operation in ANSWER_OPERATIONS:
            return final_answer(self.answers.last_read(session_id) if self.answers is not None else
                                "an ask is answered in the owner's own message, read before you reply; no tool call "
                                "answers one")
        stray = [key for key, value in args.items() if key not in READ_KEYS and value not in (None, "", [], {})]
        written = [key for key in stray if key not in SELF_SCHEMA["parameters"]["properties"]]
        if operation not in (*SELF_OPERATIONS, "status"):
            return final_answer(READS_ONLY) if written else _error(
                f"unknown operation: operation is one of {', '.join(SELF_OPERATIONS)}")
        reading = operation in {"state", "status", "log", "why", "opinions", "asks"}
        if reading and written:
            return final_answer(READS_ONLY)
        answer = self._self(operation, args, session_id)
        return _noted(answer) if reading and stray else answer

    def _self(self, operation: str, args: dict[str, Any], session_id: str) -> str:
        if operation in {"state", "status"}:
            detail = mind_state(self.client) or {}
            mind = self.settings.mind()
            switches = {"enabled": mind.get("enabled", True) is not False and detail.get("enabled") is not False,
                        "autonomy": detail.get("autonomy") or mind.get("autonomy", "standard"),
                        "sidecar_reachable": self.client.health() is not None}
            if not self._owners_own(session_id):
                # The record is the owner's, the agent's feelings included (they quote the owner's
                # obligations and reports): any other session gets the switches only.
                return _json(switches)
            narrative = self.client.narrative() or {}
            # Open asks in words: a code is never the model's to show (the owner answers in their own words).
            detail = {**detail, "asks": [{key: ask[key] for key in ("id", "title", "kind", "message") if key in ask}
                                         for ask in detail.get("asks") or [] if isinstance(ask, dict)]}
            return _json({"mind_routes": self.client.has_mind_routes() is True, **detail, **switches,
                          # From the record, never free generation: the tasks in flight and the narrative.
                          "working_on": self._working_on(),
                          "narrative": str(narrative.get("text") or "") if narrative.get("enabled") is True else ""})
        if operation in {"opinions", "withdraw", "reconsider"} or (operation == "why" and _opinion(args.get("id"))):
            return self._opinions(operation, args, session_id)
        if operation in {"log", "why"} and not self._owners_own(session_id):
            return final_answer(RECORD_IS_OWNERS)
        if operation in {"asks", "show"}:
            return self._asks(operation, args, session_id)
        if operation == "log":
            # Split: the agent's own actions apart from the mind's notes (the nightly consolidation, notices).
            params: dict[str, Any] = {"limit": max(1, min(int(args.get("limit") or 20), 100)), "split": "true"}
            if args.get("since_hours") is not None and str(args.get("since_hours")).strip():
                try:
                    params["since_hours"] = max(0.0, float(args["since_hours"]))
                except (TypeError, ValueError):
                    return _error("since_hours is a number of hours")
            for name in ("kind", "recipient"):
                if str(args.get(name) or "").strip():
                    params[name] = str(args[name]).strip()
            return self._mind("GET", "/v1/mind/log", params=params)
        if operation == "why":
            return self._why(str(args["id"])) if args.get("id") else _error("id is required")
        if operation == "rate":
            if not args.get("id") or args.get("verdict") not in VERDICTS:
                return _error(f"id and verdict ({'|'.join(VERDICTS)}) are required")
            if not self._owner(session_id):
                return final_answer("only the owner can rate an intention")
            return self._mind("POST", "/v1/mind/rate", json={"id": str(args["id"]), "verdict": args["verdict"]})
        if operation in {"grant", "revoke"}:
            # A standing grant is the owner's alone, never a guest's, a worker's or a cron run's: ``grant`` files
            # the one the owner asked for (``reason``: their words) as an ask the sidecar shows in full and
            # activates only on the owner's typed code; ``revoke`` narrows at once.
            info = self.sessions.get(session_id)
            if not self._owner(session_id) or info is None:
                return final_answer(f"only the owner can {operation} a standing grant, in their own conversation")
            contact = self.sessions.contact_id(session_id) or self.settings.owner_contact_id() or None
            if operation == "revoke":
                if not str(args.get("id") or "").strip():
                    return _error("id is required (the grant's id, from state)")
                return self._mind("POST", f"/v1/mind/grants/{quote(str(args['id']).strip(), safe='')}/revoke",
                                  json={"by": "owner", "contact_id": contact})
            if not isinstance(args.get("grant"), dict) or not args["grant"]:
                return _error("grant is required: {id, class, scope} and any limits")
            if not str(args.get("reason") or "").strip():
                return _error("reason is required: the owner's own words asking for this grant")
            if self.client.has_mind_routes() is not True:
                return final_answer(ROUTES_MISSING)
            try:
                response = self.client.post("/v1/mind/grants", timeout=5, json={
                    "grant": args["grant"], "reason": str(args["reason"]).strip()[:1500], "by": "owner",
                    "contact_id": contact, "session_id": session_id, "message": info.user_message[:8000]})
                detail = response.json().get("detail") if response.status_code == 422 else None
            except (SidecarUnavailable, ValueError, AttributeError):
                return final_answer("the sidecar is unreachable or answered no grant")
            if isinstance(detail, dict):     # a malformed grant is an argument the model can correct
                return _error(f"grant is malformed: {str(detail.get('message') or '')[:400]}")
            if not response.is_success:
                return final_answer(f"sidecar HTTP {response.status_code}")
            try:
                filed = response.json()
            except ValueError:
                return response.text
            return _json(_without_code(filed)) if isinstance(filed, dict) else response.text
        if operation in {"like", "dislike"}:     # the agent's own tastes, on its page about itself
            return self._page_line("self", operation, args.get("taste"), args.get("id"), session_id)
        return _error(f"unknown operation: operation is one of {', '.join(SELF_OPERATIONS)}")

    def _page_line(self, subject: str, kind: str, text: Any, drop: Any, session_id: str) -> str:
        """One of the agent's own lines on a page (``/v1/mind/pages``): ``text`` adds a line that names this
        turn, ``drop`` removes one by its id, both change one; with neither the page is shown, each line with its
        id. The sidecar decides which pages this session's person may open."""
        text, drop, info = str(text or "").strip(), str(drop or "").strip(), self.sessions.get(session_id)
        seen = {} if self._owner(session_id) else {"contact_id": self.sessions.contact_id(session_id) or ""}
        path = "/v1/mind/pages/" + quote(subject, safe="")
        if not text and not drop:
            return self._people("GET", path, None, params=seen)
        return self._people("POST", path + "/lines", None, json={
            "kind": kind, "text": text[:2000], "drop": drop[:32], "source": getattr(info, "turn_id", "") or "", **seen})

    def _asks(self, operation: str, args: dict[str, Any], session_id: str) -> str:
        """The owner's open asks in words (``asks``, the owner's own sessions only), or one of them put in front of
        the owner again (``show <n>``): the sidecar sends it as its own message in its own words, every term a yes
        grants in full, in the owner's own messaging chat after this turn's reply. The plugin never words an ask,
        and nothing here answers one."""
        if not self._owners_own(session_id) or self.answers is None:
            return final_answer("asks are the owner's and are shown only in the owner's own chat")
        if self.client.has_mind_routes() is not True:
            return final_answer(ROUTES_MISSING)
        asks = self.answers.open_asks()
        if asks is None:
            return final_answer("the sidecar is unreachable")
        if operation == "asks":
            return _json({"asks": self.answers.listed(session_id, asks),
                          "note": "the owner answers an ask in their own words; never ask for a code, and never "
                                  "word an ask yourself: show <n> sends it"})
        if self.answers.owner_chat(session_id) is None:
            return final_answer("an ask is shown only in the owner's own messaging chat, where they answer it")
        number = _opinion(args.get("id"))
        ask = self.answers.pick(session_id, number, asks) if number is not None else None
        if ask is None:
            return _error(f"id is the number of an open ask, from asks (1 to {len(asks)})") if asks else \
                final_answer("no ask is open")
        shown = self.answers.show(ask["id"])
        if shown == "shown":
            return _json({"showing": ask["ask"], "note": "it goes to the owner as its own message after your reply; "
                                                         "do not repeat it or ask for an answer yourself"})
        if shown == "closed":
            return final_answer("that ask is no longer open")
        if shown == "missing":
            return final_answer("this sidecar cannot show an ask again; the owner answers the ask's own message "
                                "where it was sent")
        return final_answer("the sidecar did not show the ask; nothing was sent")

    def _opinions(self, operation: str, args: dict[str, Any], session_id: str) -> str:
        """The agent's recorded opinions. Reads pass the session's participant, so the sidecar shows a guest,
        a worker or a cron run only the views meant for them; withdraw and reconsider need the owner's own
        interactive session and the owner's reason (the sidecar checks again)."""
        viewer, number = self.sessions.contact_id(session_id) or "", _opinion(args.get("id"))
        if operation == "opinions":
            return self._mind("GET", "/v1/mind/opinions", params={
                "contact_id": viewer, "limit": 10, "q": str(args.get("query") or "").strip()[:1000]})
        if operation == "why":
            return self._mind("GET", f"/v1/mind/opinions/{number}", params={"contact_id": viewer})
        if number is None:
            return _error("id is required (an opinion number from opinions)")
        if not self._owner(session_id):
            return final_answer(f"only the owner can {operation} an opinion")
        if not str(args.get("reason") or "").strip():
            return _error("reason is required: the owner's own words for why")
        return self._mind("POST", f"/v1/mind/opinions/{number}/{operation}", json={
            "reason": str(args["reason"]).strip()[:1500],
            "contact_id": viewer or self.settings.owner_contact_id() or None})

    def _working_on(self) -> list[dict[str, Any]]:
        """The tasks the mind has in flight (approved or dispatched), with their audit ids."""
        if self.client.has_mind_routes() is not True:
            return []
        try:
            response = self.client.get("/v1/mind/log", timeout=5,
                                       params={"status": "dispatched,approved", "kind": "task", "limit": 20})
            entries = response.json().get("entries") if response.is_success else None
        except (SidecarUnavailable, ValueError):
            return []
        return [{"id": item.get("id"), "title": item.get("title"), "status": item.get("status")}
                for item in (entries or []) if isinstance(item, dict) and item.get("id")]

    def _why(self, intention_id: str) -> str:
        """``GET /v1/mind/why/{id}``; an id the audit log does not hold gets the sidecar's refusal sentence,
        so a false premise about the agent's own actions is answered from the record."""
        if self.client.has_mind_routes() is not True:
            return final_answer(ROUTES_MISSING)
        try:
            response = self.client.get(f"/v1/mind/why/{quote(intention_id, safe='')}", timeout=5)
        except SidecarUnavailable:
            return final_answer("the sidecar is unreachable")
        if response.status_code == 404:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            message = detail.get("message") if isinstance(detail, dict) else None
            return final_answer(str(message or f"no intention {intention_id} exists in the audit log"))
        if not response.is_success:
            return final_answer(f"sidecar HTTP {response.status_code}")
        return response.text

    # -- colonymind_people ----------------------------------------------------------

    @in_session
    def people_tool(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        """``/v1/mind/people``: who, inspect and propose_link for everyone (a guest sees who someone is);
        permission, cadence and merge from the owner's session, to ``/v1/mind/people/changes`` with nothing of the
        turn's words: one that widens comes back as the owner's one-tap ask, which only their reply applies."""
        args = args if isinstance(args, dict) else {}
        operation, who = str(args.get("operation") or ""), str(args.get("contact_id") or "").strip()
        if operation not in (offered := PEOPLE_OPERATIONS if self.people_on else M4_PEOPLE_OPERATIONS):
            return _error(f"operation is one of {', '.join(offered)}")
        owner, viewer = self._owner(session_id), self.sessions.contact_id(session_id) or ""
        if operation in OWNER_PEOPLE_OPERATIONS and not owner:
            return final_answer("only the owner can change who may be contacted, cadences or merges")
        if operation != "who" and not who:
            return _error("contact_id (a name, handle or id) is required")
        if operation == "who" and not who and self.sessions.is_owner(session_id) is not True:
            # Listing everyone is the owner's (and the owner's senderless lanes'); a guest names one person.
            return _error("only the owner can list contacts; name who you mean")
        path, seen = "/v1/mind/people/" + quote(who, safe=""), {} if owner else {"contact_id": viewer}
        change = {"viewer": viewer or self.settings.owner_contact_id() or None, "by": "owner"}
        if operation in {"who", "inspect"}:
            params = {"q": who[:256], "limit": 10, **seen} if operation == "who" else seen
            return self._people("GET", "/v1/mind/people" if operation == "who" else path, owner, params=params)
        if operation == "boundary":     # the agent's own boundaries with this person, on its page about them
            return self._page_line(who, "boundary", args.get("text"), args.get("drop"), session_id)
        if operation == "propose_link":
            gateway, _, address = str(args.get("handle") or "").partition(":")
            if not gateway.strip() or not address.strip():
                return _error("handle is gateway:address")
            return self._people("POST", "/v1/mind/people/link", None, json={
                "contact_id": who, "gateway": gateway.strip(), "address": address.strip(),
                "evidence_refs": [f"session:{session_id}"[:256]], "by": viewer or ("owner" if owner else "guest")})
        changes = "/v1/mind/people/changes"
        if operation == "merge":
            drop = str(args.get("drop") or "").strip()
            return self._people("POST", changes, None, timeout=60, json={
                "operation": "merge", "contact_id": who, "drop": drop, **change}) if drop \
                else _error("drop (folded into contact_id) is required")
        if operation == "set_permission":
            if args.get("permission") not in {"never", "ask", "auto"}:
                return _error("permission is never, ask or auto")
            return self._people("POST", changes, None, json={
                "operation": "set_permission", "contact_id": who, "value": args["permission"], **change})
        try:
            minutes = int(args.get("minutes") or 0)
        except (TypeError, ValueError):
            return _error("minutes is a whole number (0 clears the cadence)")
        return self._people("POST", changes, None, json={
            "operation": "set_cadence", "contact_id": who, "minutes": minutes if minutes > 0 else None, **change})

    def _people(self, method: str, path: str, owner: bool | None, **kwargs: Any) -> str:
        """A read (``owner`` set) as the persons this viewer may see; a change (``owner`` None) as its outcome."""
        if self.client.has_mind_routes() is not True:
            return final_answer(ROUTES_MISSING)
        try:
            response = self.client.request(method, path, timeout=kwargs.pop("timeout", 5), **kwargs)
        except SidecarUnavailable:
            return final_answer("the sidecar is unreachable")
        try:
            value = response.json()
        except ValueError:
            value = {}
        if not response.is_success:
            detail = value.get("detail") if isinstance(value, dict) else None
            return final_answer(str(detail.get("message") or detail.get("code")) if isinstance(detail, dict) else
                                ROUTES_MISSING if response.status_code == 404 else
                                f"sidecar HTTP {response.status_code}")
        if owner is None:
            return _json(_without_code({key: value[key] for key in ("ok", "text", "status", "may_contact",
                                                                   "cadence_minutes", "candidate_id", "dropped")
                                        if key in value}))
        keep = OWNER_PERSON if owner else PUBLIC_PERSON

        def person(item: dict[str, Any]) -> dict[str, Any]:
            return {key: [f"{h.get('gateway')}:{h.get('address')}" for h in item[key] or []] if key == "handles"
                    else item[key] for key in keep if key in item}

        if "contacts" in value:
            return _json([person(item) for item in value["contacts"] if isinstance(item, dict)])
        record = person(value.get("contact") or {})
        if owner and value.get("proposals"):
            record["proposals"] = [f"{p.get('gateway')}:{p.get('address')}" for p in value["proposals"]]
        return _json(record)

    # -- memory tools ----------------------------------------------------------------

    @in_session
    def memory_search(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        args = args if isinstance(args, dict) else {}
        query = str(args.get("query") or "").strip()
        if not query:
            return _error("query is required")
        contact = self.sessions.contact_id(session_id)
        if not contact:
            return final_answer("this turn has no resolved participant; answer from the message")
        timed, problem = search_time(args)     # a search that can run: a time the sidecar would refuse is corrected
        if problem:
            return _error(problem)
        try:
            response = self.client.post("/v1/host/memory/search", timeout=10, json={
                "identity": {"host_id": "hermes"}, "person_id": contact, "session_id": session_id,
                "query": query[:4096], "limit": max(1, min(int(args.get("limit") or 5), 20)), **timed})
        except SidecarUnavailable:
            return final_answer("memory search is unavailable")
        if not response.is_success:
            return final_answer(f"memory search failed (HTTP {response.status_code})")
        value = response.json()
        retrieval = value.get("retrieval") if isinstance(value.get("retrieval"), dict) else {}
        degraded = "failed" in retrieval.values()   # e.g. semantic recall down: only exact words were matched
        if not value.get("count") and not degraded:
            return final_answer("nothing retained matches; for a participant with no history there is nothing to "
                                "find, so answer from the message and the recalled context")
        found = {"content": value.get("content", ""), "count": value.get("count", 0),
                 "source_refs": value.get("source_refs", [])}
        return _json({**found, "note": "part of the search failed, so only exact words were matched; other words "
                                       "may find more"} if degraded else found)

    @in_session
    def memory_forget(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        args = args if isinstance(args, dict) else {}
        ids = [str(item) for item in (args.get("source_ids") or []) if str(item).strip()]
        if not ids:
            return _error("source_ids is required")
        if not self._owner(session_id):
            return final_answer("only the owner can forget sources")
        contact = self.sessions.contact_id(session_id)
        try:
            response = self.client.post("/v1/host/memory/curated/sources/forget", timeout=FORGET_TIMEOUT_SECONDS,
                                        json={"person_id": contact, "source_ids": list(dict.fromkeys(ids))})
        except SidecarUnavailable as error:
            if not error.delivered:
                return final_answer("nothing was removed: the sidecar is unreachable")
            return _json({"source_erased": None, "status": "unconfirmed",
                          "note": f"The removal was sent and may have completed, but its answer did not arrive "
                                  f"within {FORGET_TIMEOUT_SECONDS:g} s. Forgetting the same source_ids again "
                                  "is safe and reports the result."})
        if response.status_code in {409, 422}:
            return _json({"source_erased": False, "error": "no matching sources; use exact source_id values"})
        if response.status_code == 410:
            return final_answer("nothing was removed: forget moved")
        if not response.is_success:
            return final_answer(f"forget failed (HTTP {response.status_code})")
        return response.text

    # -- device requests ----------------------------------------------------------

    @in_session
    def device_tool(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        """Only propose/read. OS reporting, grants, answers, dispatch and results have no model tool."""
        # Recheck this turn's own sender live; task authority and cached standing grant no device access.
        info = self.sessions.get(session_id)
        platform, sender, _, _ = session_env()
        if (info is None or not info.sender_id
                or not sender or sender != info.sender_id or platform.lower() != info.platform.lower()
                or not self.sessions.sender_is_owner(info.platform, info.sender_id, timeout=2.0)):
            return final_answer("device requests need the owner's own direct conversation")
        args = args if isinstance(args, dict) else {}
        operation = args.get("operation")
        if operation == "inventory":
            return self._mind("GET", "/v1/mind/device-capabilities")
        if operation == "status" and args.get("id"):
            return self._mind("GET", "/v1/mind/device-invocations/" + quote(str(args["id"]), safe=""))
        if operation == "request" and isinstance(args.get("invocation"), dict):
            return self._mind("POST", "/v1/mind/device-invocations", json=args["invocation"])
        return _error("use inventory, status with id, or request with invocation")

    # -- colonymind_action ----------------------------------------------------------

    def _actions(self) -> dict[str, Any]:
        """The owner's declared actions that can run (one with a problem in its declaration runs nothing)."""
        declared = self.settings.mind().get("actions")
        return {str(name): spec for name, spec in (declared.items() if isinstance(declared, dict) else ())
                if not action_problem(name, spec)}

    @in_session
    def action_tool(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        """Run a declared action: its fixed command, without a shell, with the checked arguments in its ``{name}``
        words. The owner's lanes (a session whose standing is established as the owner's, a delegated child of one
        included) and a kanban task (a mind run's call the guard let through, on a grant, the owner's yes to exactly
        it, or their approval of a task when the action's class is none) run it; a guest, and a session nothing
        establishes, never does. The declaration is read again at the call, so a changed or removed action is what
        counts."""
        args = args if isinstance(args, dict) else {}
        if not os.environ.get("HERMES_KANBAN_TASK") and self.sessions.standing(session_id)[0] != "owner":
            return final_answer("the owner's declared actions run only for the owner and the owner's own tasks")
        name, declared = str(args.get("action") or ""), self.settings.mind().get("actions")
        spec = declared.get(name) if isinstance(declared, dict) else None
        problem = f"no action {name!r} is declared" if spec is None else action_problem(name, spec)
        values, why = ({}, problem) if problem else action_values(spec, args.get("args"))
        if why:
            return _error(f"an action is called with its declared arguments ({name or 'no action named'}: {why})")
        timeout = float(spec.get("timeout") or ACTION_TIMEOUT[0])
        try:
            done = subprocess.run(action_argv(spec, values), shell=False, capture_output=True, text=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL, check=False)
        except subprocess.TimeoutExpired:
            return final_answer(f"{name} ran past its {timeout:g} s and was stopped")
        except OSError as error:
            return final_answer(f"{name} could not start ({type(error).__name__})")
        return _json({"action": name, "ok": done.returncode == 0, "exit_code": done.returncode,
                      "stdout": done.stdout[-4000:], "stderr": done.stderr[-2000:]})


__all__ = ["ANSWER_OPERATIONS", "AUTOMATED_PLATFORMS", "FORGET_SCHEMA", "FORGET_TIMEOUT_SECONDS", "PEOPLE_OPERATIONS",
           "PEOPLE_SCHEMA", "SEARCH_SCHEMA", "SELF_SCHEMA", "Tools", "VERDICTS", "action_schema", "search_time"]
