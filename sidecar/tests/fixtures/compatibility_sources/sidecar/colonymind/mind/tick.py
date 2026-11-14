"""Frozen compatibility oracle for the renderer behavior under comparison."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional

from .affect import SECTION_CHARS, Affect, postponable

from .goals import DEFAULT_MAX_TURNS, Goals, goal_lines

MIND_SECTION_CHARS = 600

class Mind:

    def _self_context(self) -> str:
        """The self-context an owner session is given: the constitution and the owner (the plugin's prompt
        section), the self-narrative and the Mind section."""
        from colonymind.config import render_constitution
        identity = self.drift.identity() or {}
        parts = [render_constitution(identity)]
        owner = identity.get("owner") if isinstance(identity.get("owner"), Mapping) else {}
        if owner.get("name"):
            parts.append(f"Your owner is {owner['name']}.")
        narrative = self.narrative()
        if narrative.get("enabled") and narrative.get("text"):
            parts.append("What you know about yourself from your own record:\n" + str(narrative["text"]))
        section = self.section()
        if section:
            parts.append(section)
        return "\n\n".join(part for part in parts if part)

    def section(self, *, limit: int = MIND_SECTION_CHARS) -> str:
        """The Mind section of an owner turn's context: at most ``limit`` characters.

        Affect's notes come first but take only the room the rest leaves (at most 360 characters),
        so they never cut the open asks. The broadcast leaves out idle curiosity (an interest's
        research): in a decision it points at the optional work the consumers postpone. A question
        curiosity raised (a contradiction for the owner) stays. Stances and the turn's lesson
        ride their own sections (``colonymind-stances``, ``colonymind-lessons``), built from the
        turn's text.
        """
        if not self.enabled:
            return ""
        lines: List[str] = []
        broadcast = [concern for concern in self.broadcast()
                     if not (concern.drive == "curiosity" and concern.kind == "interest")]
        if broadcast:
            lines.append("On my mind: " + "; ".join(f"{c.summary}"[:120] for c in broadcast) + ".")
        goals = [self.goals.render(goal) for goal in self.goals.open()]
        if goals:
            lines.append("Working toward: " + "; ".join(goal_lines(goals)) + ".")
        lines.extend(self._waiting_lines(limit - len("\n".join(lines)) - (1 if lines else 0)))
        room = limit - len("\n".join(lines)) - (1 if lines else 0)
        text = "\n".join([*self.feelings.section_lines(min(SECTION_CHARS, room)), *lines])
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return text
