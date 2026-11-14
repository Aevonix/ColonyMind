"""Owner controls and internal questions, using the frozen hands wire routes.

Hermes passes session_id separately from model arguments. No argument can grant
ownership, choose answered_by, or impersonate a child asking a question.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from typing import Any
from urllib.parse import quote
from uuid import uuid4


ACTIONS = ("list", "show", "answer", "steer", "stop", "name", "revise", "told", "accept", "continue",
           "list_hands", "pending", "dispatch")
_TERMINAL = {"accepted", "done", "failed", "stopped", "lost", "superseded", "below_minimum"}
_HAND_ID = re.compile(r"h-[0-9a-f]{12}\Z")
_TASK_ID = re.compile(r"t-[0-9a-f]{12}\Z")


def response_body(response: Any) -> dict:
    """Accept the local client seam or the adapter's HTTP response."""
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    if hasattr(response, "json"):
        response = response.json()
    if hasattr(response, "model_dump"):
        response = response.model_dump(mode="json")
    if not isinstance(response, dict):
        raise ValueError("The hands controller returned an invalid response")
    return response


def _get(runtime, path, params=None):
    return response_body(runtime.client.get(path, params=params))


def _write(runtime, path, body, request_id=""):
    digest = hashlib.sha256(json.dumps([path, body], sort_keys=True).encode()).hexdigest()[:24]
    key = f"hands-tool:{request_id or uuid4().hex}:{digest}"
    result = runtime.emit(path, body, key)
    return response_body(result) if result is not None else {"queued": True}


def _require_owner(runtime, session_id):
    if (not session_id or runtime.registry.for_session(session_id) is not None
            or runtime.authorized_owner(session_id) is not True):
        raise ValueError("Hands controls require the owner's session")


def _local_rows(runtime, session_id):
    return [{"id": hand.hand_id, "task_id": hand.task_id, "parent_session": hand.parent_session_id,
             "name": hand.plan.get("name", hand.hand_id), "state": hand.status,
             "activity": hand.activity, "intent_version": hand.plan.get("intent_version", 1)}
            for hand in runtime.registry.all() if hand.parent_session_id == session_id]


def _rows(runtime, session_id, *, allow_local=True):
    try:
        response = _get(runtime, "/v1/hands/", {"session": session_id})
    except Exception:
        if not allow_local:
            raise
        return _local_rows(runtime, session_id)
    # The controller filter is not an authority check. Check every returned row.
    return [row for row in response.get("hands", []) if row.get("parent_session") == session_id]


def _resolve(runtime, session_id, target, *, allow_local=False):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("A hand name or id is required")
    if _HAND_ID.fullmatch(target):
        try:
            row = _get(runtime, f"/v1/hands/{target}")
        except Exception:
            if not allow_local:
                raise
            row = next((row for row in _local_rows(runtime, session_id) if row["id"] == target), {})
    else:
        matches = [row for row in _rows(runtime, session_id, allow_local=allow_local)
                   if row.get("name", "").casefold() == target.casefold()]
        active = [row for row in matches if row.get("state") not in _TERMINAL]
        matches = active or matches
        if len(matches) != 1:
            raise ValueError("Hand name is missing or ambiguous; use its immutable id")
        row = matches[0]
    if row.get("parent_session") != session_id:
        raise ValueError("This hand belongs to another session")
    if not _HAND_ID.fullmatch(str(row.get("id", ""))):
        raise ValueError("Invalid hand id from controller")
    return row


def _local(runtime, row, session_id):
    hand = runtime.registry.get(row["id"])
    if hand is None or hand.parent_session_id != session_id:
        raise ValueError("No executor bound to this owner session; reconcile this hand first")
    return hand


def _text(arguments, key):
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _task(runtime, row):
    task_id = str(row.get("task_id", ""))
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("Invalid task id from controller")
    return task_id


def execute_hands(arguments, *, runtime, session_id, turn_id="", request_id="") -> dict:
    """Execute a control without ever allowing an exception into Hermes."""
    try:
        _require_owner(runtime, session_id)
        if not isinstance(arguments, dict):
            raise ValueError("Hands arguments must be an object")
        action = arguments.get("action", "list")
        if action not in ACTIONS:
            raise ValueError("Unknown hands action")
        if action == "list":
            return {"hands": _rows(runtime, session_id)}
        if action == "list_hands":
            from .definitions import list_hands
            return list_hands(runtime)
        if action == "pending":
            return _get(runtime, '/v1/hands/requests/pending')
        if action == "dispatch":
            from .definitions import dispatch_request
            return dispatch_request(runtime, _text(arguments, 'task_id'), session_id, kind=arguments.get('kind'))
        row = _resolve(runtime, session_id, arguments.get("hand"), allow_local=action in {"show", "stop", "steer"})
        hand_id = row["id"]
        if action == "show":
            return row
        if action in {"stop", "steer", "continue", "answer"}:
            hand = _local(runtime, row, session_id)
        if action == "stop":
            value = runtime.stop(hand, str(arguments.get("reason") or "Stopped by owner"),
                                 calling_session=session_id)
            return value if isinstance(value, dict) else {"ok": value is not False, "hand_id": hand_id}
        if action == "steer":
            value = runtime.steer(hand, _text(arguments, "text"), calling_session=session_id)
            return value if isinstance(value, dict) else {"ok": value is not False, "hand_id": hand_id}
        if action == "continue":
            reconciled = arguments.get("reconciled") is True
            if (row.get("state") == "unknown_effect" or hand.status == "unknown_effect") and not reconciled:
                raise ValueError("Read the effect probe and continue with reconciled=true")
            value = runtime.continue_hand(hand, answer=str(arguments.get("text") or ""), reconciled=reconciled)
            return value if isinstance(value, dict) else {"ok": True, "hand_id": hand_id}
        if action == "answer":
            text = _text(arguments, "text")
            digest = _get(runtime, "/v1/hands/digest", {"session": session_id})
            questions = {item["question_id"] for item in digest.get("items", [])
                         if item.get("hand_id") == hand_id and item.get("question_id")}
            selected = arguments.get("question_id")
            if selected:
                if selected not in questions:
                    raise ValueError("Question does not belong to this hand's digest")
            elif len(questions) == 1:
                selected = next(iter(questions))
            else:
                raise ValueError("Select an open question_id from the hand's digest")
            result = _write(runtime, f"/v1/hands/questions/{quote(selected, safe='')}/answer",
                            {"answer": text, "answered_by": session_id}, request_id)
            if result.get("queued"):
                return result
            if result.get("blocking") or hand.blocked or hand.status == "awaiting_input":
                runtime.continue_hand(hand, answer=text, reconciled=False)
            else:
                runtime.steer(hand, text, calling_session=session_id)
            return result
        if action == "name":
            name = _text(arguments, "name")
            reserved = {"hands", "colonymind_hands", "colonymind_ask", *ACTIONS}
            reserved.update(str(item).casefold() for item in runtime.config.get("reserved_names", []))
            if not re.fullmatch(r"[A-Za-z0-9-]{1,24}", name) or name.casefold() in reserved:
                raise ValueError("Use 1 to 24 letters, digits or hyphens, avoiding reserved names")
            result = _write(runtime, f"/v1/hands/{hand_id}/name", {"name": name}, request_id)
            if not result.get("queued"):
                local = runtime.registry.get(hand_id)
                if local is not None and local.parent_session_id == session_id:
                    local.plan["name"] = name
                    if isinstance(local.plan.get("task"), dict):
                        local.plan["task"]["name"] = name
            return result
        task_id = _task(runtime, row)
        if action == "revise":
            body = {key: arguments[key] for key in ("goal", "criteria", "stop") if key in arguments}
            if not body.get("goal") and not body.get("criteria"):
                raise ValueError("A revised goal or criteria is required")
            return _write(runtime, f"/v1/hands/tasks/{task_id}/revise", body, request_id)
        if action == "accept":
            version = arguments.get("for_version", row["intent_version"])
            if type(version) is not int or version < 1:
                raise ValueError("for_version must be a positive integer")
            return _write(runtime, f"/v1/hands/tasks/{task_id}/accept",
                          {"hand": hand_id, "reason": _text(arguments, "reason"),
                           "for_version": version}, request_id)
        task = _get(runtime, f"/v1/hands/tasks/{task_id}")
        obligation = task.get("obligation_id")
        if not obligation or not turn_id:
            raise ValueError("Told requires an obligation and a trusted current turn reference")
        return _write(runtime, f"/v1/hands/obligations/{quote(obligation, safe='')}/told",
                      {"turn_ref": turn_id}, request_id)
    except Exception as exc:
        return {"error": str(exc) or "Hands control unavailable"}


def execute_ask(arguments, *, runtime, session_id, request_id="") -> dict:
    """A blocking question stops tools locally even if the controller is down."""
    try:
        hand = runtime.registry.for_session(session_id)
        if hand is None:
            raise ValueError("colonymind_ask is available only inside a hand")
        if not isinstance(arguments, dict):
            raise ValueError("Question arguments must be an object")
        question = _text(arguments, "question")
        if len(question) > 500:
            raise ValueError("Question must have at most 500 characters")
        if "blocking" in arguments and not isinstance(arguments["blocking"], bool):
            raise ValueError("blocking must be true or false")
        if hand.blocked:
            raise ValueError("This hand is awaiting input; finish its report")
        blocking = arguments.get("blocking", False)
        if blocking:
            hand.blocked = True
            hand.activity = "awaiting_input"
            hand.status = "awaiting_input"
        body = {"epoch": hand.epoch or runtime.epoch, "question": question,
                "assumption": str(arguments.get("assumption") or ""), "blocking": blocking}
        result = _write(runtime, f"/v1/hands/{hand.hand_id}/question", body, request_id)
        return {**result, "blocking": blocking}
    except Exception as exc:
        return {"error": str(exc) or "Question could not be recorded"}


HAND_SCHEMA = {"name": "colonymind_hands", "description": "Inspect and control your hands. Owner sessions only.",
               "parameters": {"type": "object", "properties": {
                   "action": {"type": "string", "enum": list(ACTIONS)},
                   "hand": {"type": "string", "description": "Hand name or immutable id"},
                   "task_id": {"type": "string", "description": "Queued task id for action dispatch; pending lists requests"},
                   "kind": {"type": "string", "enum": ["coding", "deep", "research", "ops", "media", "plan",
                              "synthesize", "verify", "integrate"],
                            "description": "Task kind for dispatch (default deep); the definition can only narrow its tools"},
                   "text": {"type": "string"}, "name": {"type": "string"},
                   "question_id": {"type": "string"}, "goal": {"type": "string"},
                   "criteria": {"type": "object"}, "reason": {"type": "string"},
                   "for_version": {"type": "integer", "minimum": 1},
                   "stop": {"type": "boolean"}, "reconciled": {"type": "boolean"}},
                   "required": ["action"], "additionalProperties": False}}
ASK_SCHEMA = {"name": "colonymind_ask", "description": "Ask the front an internal question about the task. Never use it to ask permission for an action: make the call itself and the runtime holds it for the owner's approval. Blocking ends this attempt's work.",
              "parameters": {"type": "object", "properties": {
                  "question": {"type": "string", "minLength": 1, "maxLength": 500},
                  "assumption": {"type": "string"}, "blocking": {"type": "boolean", "default": False}},
                  "required": ["question"], "additionalProperties": False}}


def _correlation(session_id, turn_id="", tool_call_id=""):
    """Pinned generic handlers receive only task/session ids as keyword arguments.

    model_tools._approval_observability binds these ContextVars around registry
    dispatch in the executing thread. Read them only for the same session, never
    from model arguments or a process-global environment variable.
    """
    try:
        context = importlib.import_module("tools.approval_context")
        if context._approval_session_id.get() == session_id:
            turn_id = turn_id or context._approval_turn_id.get()
            tool_call_id = tool_call_id or context._approval_tool_call_id.get()
    except Exception:
        pass
    return turn_id, tool_call_id


def register(ctx: Any) -> None:
    from .dispatch import get_runtime

    def hands(args=None, *, session_id="", turn_id="", tool_call_id="", **_):
        try:
            runtime = get_runtime(ctx)
            turn_id, tool_call_id = _correlation(session_id, turn_id, tool_call_id)
            # The actual current turn comes from the captured parent when Hermes
            # does not pass turn_id to a generic plugin handler.
            current_turn = turn_id or str(getattr(runtime, "contexts", {}).get(session_id, {}).get("turn") or "")
            result = execute_hands(args, runtime=runtime, session_id=session_id,
                                   turn_id=current_turn, request_id=tool_call_id)
        except Exception:
            result = {"error": "Hands runtime unavailable"}
        return json.dumps(result, default=str)

    def ask(args=None, *, session_id="", tool_call_id="", **_):
        try:
            _, tool_call_id = _correlation(session_id, tool_call_id=tool_call_id)
            result = execute_ask(args, runtime=get_runtime(ctx), session_id=session_id,
                                 request_id=tool_call_id)
        except Exception:
            result = {"error": "Hands runtime unavailable"}
        return json.dumps(result, default=str)

    for schema, handler in ((HAND_SCHEMA, hands), (ASK_SCHEMA, ask)):
        ctx.register_tool(name=schema["name"], toolset="colonymind-hands", schema=schema, handler=handler)
