"""The /hands command uses Hermes' normal between-turn command path."""

from __future__ import annotations

from contextvars import ContextVar
import json
import re
import shlex
from typing import Any

from .tools import ACTIONS, execute_hands


USAGE = ("/hands [show NAME | stop NAME | stop all | steer NAME TEXT | answer NAME TEXT | "
         "name NAME NEW-NAME | revise NAME GOAL | told NAME | accept NAME REASON | "
         "continue NAME [--reconciled]]. /hands runs between turns; /stop interrupts the current turn.")
_COMMAND = re.compile(r"^/hands(?:\s|$)", re.IGNORECASE)
_binding: ContextVar[tuple[str, str]] = ContextVar("colonymind_hands_command", default=("", ""))


def run(raw_args: str, *, runtime, session_id: str, turn_id: str = "") -> str:
    try:
        words = shlex.split(raw_args or "")
        if not words:
            arguments = {"action": "list"}
        else:
            action = words[0].lower()
            if action not in ACTIONS:
                return USAGE
            arguments = {"action": action}
            if len(words) > 1:
                arguments["hand"] = words[1]
            if action == "stop" and arguments.get("hand") == "all":
                listed = execute_hands({"action": "list"}, runtime=runtime, session_id=session_id)
                if "error" in listed:
                    return "Error: " + listed["error"]
                results = [execute_hands({"action": "stop", "hand": row["id"]}, runtime=runtime,
                                         session_id=session_id) for row in listed.get("hands", [])
                           if row.get("state") in {"planned", "queued", "running", "checking", "awaiting_input",
                                                   "awaiting_approval"}]
                return json.dumps({"stopped": results}, default=str)
            if action in {"steer", "answer"}:
                arguments["text"] = " ".join(words[2:])
            elif action == "name":
                arguments["name"] = " ".join(words[2:])
            elif action == "revise":
                arguments["goal"] = " ".join(words[2:])
            elif action in {"stop", "accept"} and len(words) > 2:
                arguments["reason"] = " ".join(words[2:])
            elif action == "continue":
                arguments["reconciled"] = "--reconciled" in words[2:]
        result = execute_hands(arguments, runtime=runtime, session_id=session_id, turn_id=turn_id)
        if "error" in result:
            return "Error: " + result["error"]
        if "hands" in result:
            rows = result["hands"]
            return "\n".join(f"{row.get('name', row['id'])}: {row.get('state', 'unknown')} ({row['id']})"
                             for row in rows) or "No hands in this session."
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return "Hands command unavailable. " + USAGE


def command_session(event, gateway, runtime) -> str:
    """Resolve the sender and stored session before binding command authority.

    The gateway invokes this hook before auth and before its current-session
    ContextVars are populated. A prior ContextVar is never proof of ownership.
    """
    source = getattr(event, "source", None)
    sessions = getattr(runtime, "sessions", None)
    if source is None or sessions is None:
        return ""
    platform = getattr(source, "platform", "")
    platform = str(getattr(platform, "value", platform))
    sender = str(getattr(source, "user_id", "") or "")
    if sessions.sender_is_owner(platform, sender) is not True:
        return ""
    key = gateway._session_key_for_source(source)
    session_id = gateway.session_store.peek_session_id(key)
    if not isinstance(session_id, str) or not session_id:
        return ""
    return session_id if runtime.authorized_owner(session_id) is True else ""


def register(ctx: Any) -> None:
    from .dispatch import get_runtime

    def gate(event=None, gateway=None, **_):
        _binding.set(("", ""))
        if not _COMMAND.match(str(getattr(event, "text", "") or "")):
            return None
        try:
            session_id = command_session(event, gateway, get_runtime(ctx))
            if session_id:
                _binding.set((session_id, str(getattr(event, "message_id", "") or "")))
                return None
        except Exception:
            pass
        return {"action": "skip", "reason": "Hands commands require an authenticated owner session"}

    def handler(raw_args=""):
        session_id, turn_id = _binding.get()
        _binding.set(("", ""))
        try:
            return run(raw_args, runtime=get_runtime(ctx), session_id=session_id, turn_id=turn_id)
        except Exception:
            return "Hands runtime unavailable."

    ctx.register_hook("pre_gateway_dispatch", gate)
    ctx.register_command("hands", handler, description="Inspect and control hands between turns",
                         args_hint="[show NAME | stop NAME | stop all]", argument_mode="text")
