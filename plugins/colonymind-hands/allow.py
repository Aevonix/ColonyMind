"""Named tool policy for hand sessions; shell-capable hands remain trusted lanes."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .budgets import WAITING, TERMINAL, block, ended_message, is_child_context, refuse, resolve_hand

logger = logging.getLogger(__name__)
SHELL_KINDS = frozenset({"coding", "ops", "integrate"})
READ_TOOLS = frozenset({"read_file", "search_files", "list_directory", "colonymind_memory_search", "colonymind_ask", "skills_list", "skill_view"})
ALLOW = {
    "research": READ_TOOLS | {"web_search", "web_extract"},
    "deep": READ_TOOLS, "plan": READ_TOOLS, "synthesize": READ_TOOLS, "verify": READ_TOOLS,
    "media": READ_TOOLS | {"vision_analyze", "image_generate", "write_file", "edit_file"},
}
BLOCKED = frozenset({"send_message", "clarify", "delegate_task", "cronjob", "schedule_cronjob", "kanban", "computer_use",
    "cronjob_manage", "memory", "colonymind_memory_forget", "colonymind_people", "colonymind_secret", "colonymind_secrets",
    "colonymind_self", "colonymind_reminder", "colonymind_hands", "colonymind_task", "colonymind_action"})


def check_tool(hand: Any, tool_name: str, args: dict[str, Any]) -> str | None:
    if hand.blocked or hand.activity in WAITING or hand.status in TERMINAL:
        return ended_message(hand)
    if tool_name in BLOCKED or tool_name.startswith(("colonymind_secret", "colonymind_people", "colonymind_memory_write")):
        return "this tool is unavailable in hand sessions"
    if ('definition_policy' in hand.plan or 'definition_snapshot' in hand.plan
            or hand.plan.get('task', {}).get('hand')):
        from .definitions import check_policy
        restricted = check_policy(hand, tool_name, args)
        if restricted:
            return restricted
    if hand.kind in SHELL_KINDS:
        return None
    if tool_name not in ALLOW.get(hand.kind, frozenset()):
        return f"{tool_name} is outside this hand's allow-list"
    if hand.kind == "media" and tool_name in {"write_file", "edit_file", "image_generate"}:
        workspace = hand.plan.get("workspace") or hand.plan.get("workdir") or hand.plan.get("cwd")
        target = args.get("path") or args.get("output_path")
        if not workspace or not target:
            return "media writes require a path in the hand workspace"
        root = Path(workspace).resolve()
        path = Path(target)
        resolved = (path if path.is_absolute() else root / path).resolve()
        if not resolved.is_relative_to(root):
            return "media writes must remain in the hand workspace"
    return None


def pre_tool_call(runtime: Any, tool_name: str, args: dict[str, Any] | None = None,
                  session_id: str = "", **ids: Any) -> dict | None:
    hand = None
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if hand is None:
            return block("unbound hand session: tool refused") if is_child_context(**ids) else None
        with hand.lock:
            reason = check_tool(hand, tool_name, args or {})
        return refuse(hand, reason, ids.get("tool_call_id", "")) if reason else None
    except Exception:
        logger.exception("hand allow-list check failed")
        if hand is None and not is_child_context(**ids):
            return None
        return refuse(hand, "hand allow-list check unavailable: tool refused", ids.get("tool_call_id", ""))


def register(ctx: Any) -> None:
    from .dispatch import get_runtime

    def before(**kwargs: Any) -> dict | None:
        try:
            return pre_tool_call(get_runtime(ctx), **kwargs)
        except Exception:
            logger.exception("hand allow-list runtime unavailable")
            return block("hand allow-list check unavailable: tool refused") if is_child_context(**kwargs) else None

    ctx.register_hook("pre_tool_call", before)
