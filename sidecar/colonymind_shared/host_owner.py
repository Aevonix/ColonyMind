"""Whether the owner of the host wrote a turn, as Hermes 0.22 identifies that owner.

Hermes 0.22 runs every local session as the operating-system user who runs Hermes and gives that user one
subject. A turn is the owner's when that subject sent it in a direct chat on one of Hermes' own chat surfaces
and Hermes names nobody else as its author. The memory provider and the capture plugin both decide with
``decide``, each from what Hermes handed it, so the two cannot disagree. Every doubt answers no. Hermes 0.21
has no such subject, so nothing here answers yes there.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

# The labels of the chats the owner opens: the terminal, the TUI and the desktop app. Editor clients, hosted
# rooms, forwarded agent-to-agent turns, scheduled runs and task workers carry other labels.
OWN_CHAT_SURFACES = frozenset({"cli", "tui", "desktop"})
_BOUND = ("PLATFORM", "USER_ID", "CHAT_ID", "CHAT_TYPE")
_STARTED = ("_user_id", "_chat_id", "_chat_type")


def subject() -> str:
    """Hermes' subject for the host's owner; "" where Hermes names none or it cannot be read."""
    try:
        from gateway.owner_identity import host_owner_subject
        value = host_owner_subject()
    except Exception:
        return ""
    return value if isinstance(value, str) else ""


def bound_identity() -> tuple[str, str, str, str]:
    """``(platform, user, chat, chat type)`` the gateway bound for the running turn; blanks without one."""
    try:
        from gateway.session_context import get_session_env
        return tuple(get_session_env("HERMES_SESSION_" + key, "") or "" for key in _BOUND)  # type: ignore[return-value]
    except Exception:
        return "", "", "", ""


def named_author(raw: Any) -> Optional[tuple[str, bool]]:
    """``(id, is a bot)`` of the author Hermes names for a turn (its ``{"id", "name", "is_bot"}``); None when it
    names none."""
    if not isinstance(raw, Mapping):
        return None
    author_id, name, bot = raw.get("id"), raw.get("name"), raw.get("is_bot")
    if not author_id and not name and not bot:
        return None
    return (author_id if isinstance(author_id, str) else ""), bool(bot)


def decide(*, label: Any, started: tuple, bound: tuple, hook: Optional[tuple] = None,
           author: Optional[tuple] = None) -> bool:
    """Whether the host's owner wrote the turn.

    ``label`` is the surface Hermes built the agent for, ``started`` the ``(user, chat, chat type)`` it built the
    agent with, ``bound`` the gateway's identity of this turn (blanks without a gateway), ``hook`` the
    ``(platform, user)`` the turn's ``pre_llm_call`` named when one ran, ``author`` the turn's ``named_author``.

    A gateway identity decides alone: a local direct chat of the owner's subject, on one of the owner's own chat
    surfaces. Without one, only a terminal the agent was built for counts, and a hook has to agree with it. On the
    owner's own surface a turn without a named author is taken as the owner's; an author Hermes names must be the
    owner and no bot, so a message whose named author is a bot, a peer or a dispatcher is not the owner's. A task
    run never is.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    try:
        if any(bound):
            platform, sender, chat, chat_type = bound
            if platform != "local" or label not in OWN_CHAT_SURFACES:
                return False
        else:
            sender, chat, chat_type = started
            if label != "cli" or (hook is not None and tuple(hook) != (label, sender)):
                return False
        if not sender or not chat or chat_type != "dm":
            return False
        owner = subject()
        if not owner or sender != owner:
            return False
        return author is None or (not author[1] and author[0] == owner)
    except Exception:
        return False


def running_turn(session_id: str = "", hook: Optional[tuple] = None) -> bool:
    """``decide`` for the turn this thread runs, from the agent Hermes bound for it: its surface, the identity it
    was built with and the author Hermes set on it for this turn. No bound agent, or one of another session,
    answers no."""
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
        agent = get_active_subagent_parent()
        if agent is None or (session_id and getattr(agent, "session_id", None) != session_id):
            return False
        return decide(label=getattr(agent, "platform", None) or "",
                      started=tuple(getattr(agent, name, None) or "" for name in _STARTED),
                      bound=bound_identity(), hook=hook, author=named_author(getattr(agent, "_turn_author", None)))
    except Exception:
        return False


__all__ = ["OWN_CHAT_SURFACES", "bound_identity", "decide", "named_author", "running_turn", "subject"]
